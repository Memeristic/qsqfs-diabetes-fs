function [selected, bestFitness, history] = QSQ_FS(X, y, params)
% QSQ_FS  Quorum Sensing & Quorum Quenching feature selection.
%
%   [selected, bestFitness, history] = QSQ_FS(X, y, params)
%
%   Reference MATLAB implementation, kept in parity with the Python engine
%   (src/qsfs.py). Key mechanisms:
%     * EMA smoothing of the autoinducer field (Eq 3.7a).
%     * Per-position guided refinement in Stage 2 (gap*rho), rather than a
%       single global blend factor (Sec 3.5.2).
%     * Elite preservation: the global-best colony is re-injected each
%       generation.
%     * QQ suppression computed from raw (un-penalised) fitness (Eq 3.8), so the
%       effective fitness of thesis Sec 3.5.3 is F_eff = F_raw - suppression.
%     * rng(params.seed) for reproducibility.
%     * Ablation switches useQS / useQQ / useCache / elitism (all default true =
%       full algorithm), matching src/qsfs.py so matlab/ablation_study.m can
%       disable each mechanism independently.
%
%   INPUTS
%     X       : n-by-d numeric feature matrix
%     y       : n-by-1 binary label vector (0/1)
%     params  : struct with optional fields (defaults in getDefaults below):
%               nColonies, maxIter1, maxIter2, alpha, wAI, delta1, delta2,
%               rho, emaBeta, kNN, cvFolds, mutationRate1, mutationRate2,
%               injectMutRate, injectWAI, recoveryIters, stagnationWindow,
%               diversityThresh, seed, verbose, useQS, useQQ, useCache, elitism,
%               featurePool, poolWeights
%
%     featurePool / poolWeights (optional) : When supplied, Stage 2 is seeded by
%               frequency-weighted sampling from `featurePool` (a vector of
%               GLOBAL column indices into X, e.g. the Stage-1 candidate
%               features) with per-feature `poolWeights` (e.g. cross-modality
%               selection counts) -- exactly mirroring the Python engine's
%               QSQFS.fit(X, y, feature_pool=..., pool_weights=...) in
%               src/qsfs.py. Call QSQ_FS on the FULL matrix X (not X(:,pool));
%               mutation can still explore outside the pool, so the returned
%               indices are global. When omitted, Stage 2 derives its pool from
%               the Stage-1 population as before (unchanged behaviour).
%
%   OUTPUTS
%     selected     : indices of selected feature columns
%     bestFitness  : best objective value found (Eq 3.1)
%     history      : struct with .fitness and .featureCount per iteration
%
%   NOTE: fitness uses stratified k-fold KNN with per-fold standardisation
%   (scaler fit on the training fold only) -- the no-leakage micro-pattern.

    if nargin < 3, params = struct(); end
    p = getDefaults(params);
    rng(p.seed);                                   % reproducibility

    [n, d] = size(X);
    nMin = max(1, ceil(0.05 * d));                 % Eq 3.2
    nMax = ceil(0.50 * d);

    ctx = struct('X', X, 'y', y, 'kNN', p.kNN, 'cvFolds', p.cvFolds, ...
                 'alpha', p.alpha, 'd', d, 'seed', p.seed);
    cache = containers.Map('KeyType', 'char', 'ValueType', 'double');

    histFit = []; histCnt = [];

    %% ---------------- Stage 1 ----------------
    pop = initPopulation(p.nColonies, d, nMin, nMax);
    [fit, cache] = evalPop(pop, ctx, cache, containers.Map('KeyType','char','ValueType','double'), p);
    [gBestFit, gi] = max(fit); gBest = pop(gi, :);
    archive = containers.Map('KeyType', 'char', 'ValueType', 'double');
    aiHist = [];
    wAI = p.wAI; injectActive = false; recoveryLeft = 0;

    for it = 1:p.maxIter1
        thetaS = mean(fit);
        strongIdx = find(fit >= thetaS);
        weakIdx   = find(fit <  p.weakThresh1);

        [ai, aiHist] = computeAI(pop, fit, strongIdx, wAI, p.emaBeta, aiHist, d, it, p.seed, p.useQS);
        if ~isempty(strongIdx)
            [~, sb] = max(fit(strongIdx)); strongBest = pop(strongIdx(sb), :);
        else
            strongBest = gBest;
        end

        for i = weakIdx(:)'
            pop(i, :) = mutate(pop(i, :), ai, strongBest, fit(i), wAI, ...
                               pickMutRate(injectActive, p.mutationRate1, p.injectMutRate), ...
                               d, nMin, nMax, p.seed, it, i);
        end

        if p.useQQ
            archive = updateArchive(archive, pop, weakIdx, thetaS, p.delta1, ctx, cache, p);
        end
        [fit, cache] = evalPop(pop, ctx, cache, archive, p);

        % elitism (ablation switch p.elitism): re-inject best, then refresh global best
        if p.elitism
            [wf, wi] = min(fit);
            if ~isequal(pop(wi, :), gBest), pop(wi, :) = gBest; fit(wi) = gBestFit; end
        end
        [mx, mi] = max(fit);
        if mx > gBestFit, gBestFit = mx; gBest = pop(mi, :); end

        histFit(end+1) = gBestFit; histCnt(end+1) = sum(gBest); %#ok<AGROW>

        [injectActive, recoveryLeft, wAI, pop] = checkStagnation( ...
            histFit, p, injectActive, recoveryLeft, wAI, pop, fit, gBest, nMin, nMax);

        if p.verbose && mod(it, 5) == 0
            fprintf('Stage1 it=%3d  best=%.4f  feats=%d  arch=%d\n', ...
                    it, gBestFit, sum(gBest), archive.Count);
        end
        if std(fit) < p.diversityThresh, break; end
    end
    stage1Boundary = numel(histFit);

    %% ---------------- Stage 2 ----------------
    % [F] Frequency-weighted Stage-2 initialisation (thesis Sec 3.5.1).
    % If the caller supplies a feature pool + weights (nested_cv_pipeline passes
    % the Stage-1 candidate features and their cross-modality selection counts),
    % seed Stage 2 from THAT pool -- indices are GLOBAL into the full d-column X,
    % exactly mirroring src/qsfs.py's fit(..., feature_pool=, pool_weights=)
    % branch. Otherwise derive the pool from the Stage-1 population as before.
    % In both cases the mutation operators below can still flip any of the d
    % bits, so exploration outside the pool is preserved (the pool only biases
    % the initial seed) -- this is what makes it match the Python engine instead
    % of merely column-restricting.
    if isfield(p, 'featurePool') && ~isempty(p.featurePool)
        pool = p.featurePool(:)';
        pool = pool(pool >= 1 & pool <= d);          % guard against stray indices
        if isempty(pool), pool = 1:d; end
        if isfield(p, 'poolWeights') && numel(p.poolWeights) == numel(p.featurePool)
            w_all = p.poolWeights(:)';
            weights = w_all(p.featurePool(:)' >= 1 & p.featurePool(:)' <= d);
        else
            weights = ones(1, numel(pool));
        end
    else
        freq = mean(pop(max(fit) == fit, :), 1);
        pool = find(freq > 0.3); if isempty(pool), pool = 1:d; end
        weights = freq(pool);
    end
    if sum(weights) > 0, weights = weights / sum(weights); end

    archive = containers.Map('KeyType', 'char', 'ValueType', 'double');
    aiHist = []; wAI = p.wAI; injectActive = false;
    for i = 1:p.nColonies
        k = max(1, ceil(0.30 * numel(pool)));
        chosen = weightedSample(pool, weights, k, p.seed + i);
        col = zeros(1, d); col(chosen) = 1;
        pop(i, :) = repair(col, nMin, nMax, p.seed + 1000 + i);
    end
    [fit, cache] = evalPop(pop, ctx, cache, archive, p);
    [mx, mi] = max(fit); if mx > gBestFit, gBestFit = mx; gBest = pop(mi, :); end

    for it = 1:p.maxIter2
        thetaS = prctile(fit, 85);
        thetaW = mean(fit);
        strongIdx = find(fit >= thetaS);
        weakIdx   = find(fit <  thetaW);

        [ai, aiHist] = computeAI(pop, fit, strongIdx, wAI, p.emaBeta, aiHist, d, it, p.seed + 7, p.useQS);
        if ~isempty(strongIdx)
            [~, sb] = max(fit(strongIdx)); strongBest = pop(strongIdx(sb), :);
        else
            strongBest = gBest;
        end

        for i = weakIdx(:)'
            m = mutate(pop(i, :), ai, strongBest, fit(i), wAI, ...
                       pickMutRate(injectActive, p.mutationRate2, p.injectMutRate), ...
                       d, nMin, nMax, p.seed, it + 1000, i);
            % per-position guided refinement (Sec 3.5.2): blend toward global best
            gap = max(0, gBestFit - fit(i));
            rg  = localRng(p.seed + 97 * it + i);
            blend = rand(rg, 1, d) < gap * p.rho;
            m(blend) = gBest(blend);
            pop(i, :) = repair(m, nMin, nMax, p.seed + 2000 + i);
        end

        if p.useQQ
            archive = updateArchive(archive, pop, weakIdx, thetaS, p.delta2, ctx, cache, p);
        end
        [fit, cache] = evalPop(pop, ctx, cache, archive, p);

        if p.elitism
            [wf, wi] = min(fit);
            if ~isequal(pop(wi, :), gBest), pop(wi, :) = gBest; fit(wi) = gBestFit; end
        end
        [mx, mi] = max(fit);
        if mx > gBestFit, gBestFit = mx; gBest = pop(mi, :); end

        histFit(end+1) = gBestFit; histCnt(end+1) = sum(gBest); %#ok<AGROW>
        if p.verbose && mod(it, 10) == 0
            fprintf('Stage2 it=%3d  best=%.4f  feats=%d\n', it, gBestFit, sum(gBest));
        end
        if std(fit) < 0.005, break; end
    end

    selected = find(gBest == 1);
    bestFitness = gBestFit;
    history = struct('fitness', histFit, 'featureCount', histCnt, ...
                     'stage1Boundary', stage1Boundary);
end

% ======================================================================= %
function p = getDefaults(params)
    d.nColonies = 50; d.maxIter1 = 30; d.maxIter2 = 70;
    d.alpha = 0.95; d.wAI = 0.50; d.delta1 = 0.97; d.delta2 = 0.95;
    d.rho = 0.80; d.emaBeta = 0.70; d.kNN = 3; d.cvFolds = 10;
    d.mutationRate1 = 0.15; d.mutationRate2 = 0.25;
    d.injectMutRate = 0.40; d.injectWAI = 0.20; d.recoveryIters = 5;
    d.stagnationWindow = 15; d.diversityThresh = 0.05; d.weakThresh1 = 0.30;
    d.seed = 42; d.verbose = true;
    % Ablation switches (thesis Ch.5, in parity with src/qsfs.py):
    % all default TRUE = full algorithm.
    d.useQS = true;      % Quorum Sensing (autoinducer propagation)
    d.useQQ = true;      % Quorum Quenching (suppression archive)
    d.useCache = true;   % fitness memoisation (runtime only, no effect on values)
    d.elitism = true;    % re-inject global-best colony each generation
    % Optional Stage-2 pool seeding (empty = derive pool from Stage-1 population):
    d.featurePool = [];  % global column indices to seed Stage 2 from
    d.poolWeights = [];  % per-featurePool sampling weights (e.g. selection counts)
    f = fieldnames(d);
    for i = 1:numel(f)
        if ~isfield(params, f{i}), params.(f{i}) = d.(f{i}); end
    end
    p = params;
end

function r = localRng(seed)
    r = RandStream('mt19937ar', 'Seed', mod(seed, 2^32));
end

function mr = pickMutRate(injectActive, base, inj)
    if injectActive, mr = inj; else, mr = base; end
end

function pop = initPopulation(nC, d, nMin, nMax)
    pop = zeros(nC, d);
    for i = 1:nC
        pop(i, :) = repair(double(rand(1, d) > 0.5), nMin, nMax, i);
    end
end

function c = repair(c, nMin, nMax, seed)
    rg = localRng(seed);
    s = sum(c);
    if s < nMin
        z = find(c == 0); need = nMin - s;
        if numel(z) >= need
            idx = z(randperm(rg, numel(z), need)); c(idx) = 1;
        end
    elseif s > nMax
        o = find(c == 1); surplus = s - nMax;
        if numel(o) >= surplus
            idx = o(randperm(rg, numel(o), surplus)); c(idx) = 0;
        end
    end
end

function [fit, cache] = evalPop(pop, ctx, cache, archive, p)
    % Effective Fitness (thesis Sec 3.5.3, Eq 3.8): the Quorum Quenching archive
    % suppression is subtracted from the raw fitness,
    % F_eff(c) = F_raw(c) - suppression(c), matching src/qsfs.py's
    % `_evaluate_population` (fitness = raw - penalties).
    n = size(pop, 1); fit = zeros(1, n);
    for i = 1:n
        key = char('0' + pop(i, :));
        if p.useCache && isKey(cache, key)
            raw = cache(key);
        else
            raw = fitnessKNN(pop(i, :), ctx);
            if p.useCache, cache(key) = raw; end
        end
        if p.useQQ && isKey(archive, key)
            penalty = archive(key);
        else
            penalty = 0;
        end
        fit(i) = raw - penalty;
    end
end

function f = fitnessKNN(mask, ctx)
    sel = find(mask == 1);
    if isempty(sel), f = 0.5; return; end
    Xs = ctx.X(:, sel); y = ctx.y;
    cv = cvpartition(y, 'KFold', ctx.cvFolds, 'Stratify', true);
    acc = zeros(cv.NumTestSets, 1);
    for k = 1:cv.NumTestSets
        tr = training(cv, k); te = test(cv, k);
        mu = mean(Xs(tr, :), 1); sd = std(Xs(tr, :), 0, 1) + 1e-10;   % fit on train fold
        Xtr = (Xs(tr, :) - mu) ./ sd; Xte = (Xs(te, :) - mu) ./ sd;
        mdl = fitcknn(Xtr, y(tr), 'NumNeighbors', ctx.kNN);
        acc(k) = mean(predict(mdl, Xte) == y(te));
    end
    parsimony = 1 - numel(sel) / ctx.d;
    f = ctx.alpha * mean(acc) + (1 - ctx.alpha) * parsimony;          % Eq 3.1
end

function [ai, aiHist] = computeAI(pop, fit, strongIdx, wAI, beta, aiHist, d, it, seed, useQS)
    % Ablation switch: useQS=false removes exploitation pressure entirely
    % (mirrors src/qsfs.py's "No QS (w_AI off)" variant), leaving only the
    % random background term.
    if nargin < 10, useQS = true; end
    if ~useQS
        rg = localRng(seed + it); ai = rand(rg, 1, d) * 0.1; return;
    end
    if isempty(strongIdx)
        rg = localRng(seed + it); ai = rand(rg, 1, d) * 0.1; return;
    end
    sf = fit(strongIdx);
    nrm = (sf - min(sf)) / (max(sf) - min(sf) + 1e-10);               % Eq 3.5
    newAI = zeros(1, d); rg = localRng(seed + it);
    for j = 1:numel(strongIdx)
        c = pop(strongIdx(j), :);
        U = rand(rg, 1, d);
        contrib = c .* (nrm(j) * wAI + (1 - wAI) * U);                % Eq 3.6
        newAI = max(newAI, contrib);                                 % Eq 3.7
    end
    if isempty(aiHist)
        aiHist = newAI;
    else
        aiHist = beta * newAI + (1 - beta) * aiHist;                 % Eq 3.7a EMA
    end
    ai = aiHist;
end

function m = mutate(c, ai, strongBest, fc, wAI, mutRate, d, nMin, nMax, seed, it, idx)
    rg = localRng(seed * 1000003 + it * 1009 + idx);
    m = c; U1 = rand(rg, 1, d); U2 = rand(rg, 1, d); U0 = rand(rg, 1, d);
    for j = 1:d
        if U1(j) < ai(j)              % Eq 3.10 inherit
            m(j) = strongBest(j);
        elseif U2(j) < fc * wAI       % Eq 3.11 retain
            % keep
        else                          % Eq 3.12 explore
            m(j) = double(rand(rg) > 0.5);
        end
        if U0(j) < mutRate            % background mutation (Sec 3.4.3 mech 1)
            m(j) = double(rand(rg) > 0.5);
        end
    end
    m = repair(m, nMin, nMax, seed + idx);
end

function archive = updateArchive(archive, pop, weakIdx, thetaS, delta, ctx, cache, p)
    ks = keys(archive);                                              % Eq 3.9 decay
    for i = 1:numel(ks)
        archive(ks{i}) = archive(ks{i}) * delta;
        if archive(ks{i}) < 1e-3, remove(archive, ks{i}); end
    end
    for i = weakIdx(:)'
        key = char('0' + pop(i, :));
        if p.useCache && isKey(cache, key), raw = cache(key); else, raw = fitnessKNN(pop(i, :), ctx); end
        supp = max(0, thetaS - raw);                                 % Eq 3.8 (RAW fitness)
        if ~isKey(archive, key) || archive(key) < supp, archive(key) = supp; end
    end
end

function [injectActive, recoveryLeft, wAI, pop] = checkStagnation( ...
        histFit, p, injectActive, recoveryLeft, wAI, pop, fit, gBest, nMin, nMax)
    w = p.stagnationWindow;
    if numel(histFit) < w, return; end
    window = histFit(end - w + 1:end);
    stagnating = window(end) <= window(1) + 1e-9;
    if stagnating && ~injectActive
        injectActive = true; recoveryLeft = p.recoveryIters; wAI = p.injectWAI;
        nRep = max(1, floor(0.25 * size(pop, 1)));
        [~, order] = sort(fit, 'ascend'); rg = localRng(p.seed + 7919);
        for k = 1:nRep
            idx = order(k);
            if isequal(pop(idx, :), gBest), continue; end
            pop(idx, :) = repair(double(rand(rg, 1, size(pop, 2)) > 0.5), nMin, nMax, idx);
        end
    elseif injectActive
        recoveryLeft = recoveryLeft - 1;
        if ~stagnating || recoveryLeft <= 0
            injectActive = false; wAI = p.wAI;
        end
    end
end

function chosen = weightedSample(pool, weights, k, seed)
    rg = localRng(seed);
    if isempty(weights) || sum(weights) == 0
        idx = randperm(rg, numel(pool), min(k, numel(pool)));
        chosen = pool(idx); return;
    end
    chosen = zeros(1, 0); avail = pool; w = weights;
    for i = 1:min(k, numel(pool))
        c = cumsum(w) / sum(w); r = rand(rg);
        j = find(c >= r, 1, 'first'); if isempty(j), j = numel(avail); end
        chosen(end+1) = avail(j); %#ok<AGROW>
        avail(j) = []; w(j) = [];
    end
end
