function sel = feature_selectors(method, X, y, nAgents, maxIter, seed, kNN, cvFolds, metric)
% FEATURE_SELECTORS  Binary metaheuristic feature selectors (RIME, PLO, HGS,
% GA, PSO), the MATLAB counterparts of src/optimizers.py.
%
%   sel = feature_selectors(method, X, y, nAgents, maxIter, seed, kNN, cvFolds, metric)
%
% These are the proposal's named algorithms -- RIME (Rime optimisation), PLO
% (Polar Lights Optimizer) and HGS (Hunger Games Search) -- plus GA and PSO,
% all as first-class feature selectors sharing one leak-free wrapper fitness so
% an equal population/iteration budget is a fair comparison against QSQ_FS.m.
%
% INPUTS
%   method  : 'RIME' | 'PLO' | 'HGS' | 'GA' | 'PSO' (case-insensitive)
%   X, y    : n-by-d feature matrix / binary label vector (0/1)
%   nAgents : population size          (default 30)
%   maxIter : iterations               (default 20)
%   seed    : RNG seed                 (default 42)
%   kNN     : neighbours in the fitness KNN (default 3)
%   cvFolds : folds in the fitness CV  (default 3)
%   metric  : 'accuracy' | 'auc' | 'balanced' (default 'accuracy')
%
% OUTPUT
%   sel     : indices of selected feature columns
%
% Portable to base MATLAB and GNU Octave (manual KNN CV, no toolbox calls).

    if nargin < 4 || isempty(nAgents), nAgents = 30; end
    if nargin < 5 || isempty(maxIter), maxIter = 20; end
    if nargin < 6 || isempty(seed), seed = 42; end
    if nargin < 7 || isempty(kNN), kNN = 3; end
    if nargin < 8 || isempty(cvFolds), cvFolds = 3; end
    if nargin < 9 || isempty(metric), metric = 'accuracy'; end
    rng(seed, 'twister');

    d = size(X, 2);
    ctx = struct('X', X, 'y', y, 'kNN', kNN, 'cvFolds', cvFolds, ...
                 'metric', metric, 'alpha', 0.95, 'd', d);

    switch upper(method)
        case 'RIME', best = runRIME(ctx, nAgents, maxIter);
        case 'PLO',  best = runPLO(ctx, nAgents, maxIter);
        case 'HGS',  best = runHGS(ctx, nAgents, maxIter);
        case 'GA',   best = runGA(ctx, nAgents, maxIter);
        case 'PSO',  best = runPSO(ctx, nAgents, maxIter);
        otherwise
            error('feature_selectors:badMethod', ...
                  'Unknown method "%s". Use RIME|PLO|HGS|GA|PSO.', method);
    end
    sel = find(best == 1);
end

% ======================================================================= %
function [pop, fit, best, bestFit] = initPop(ctx, nA)
    d = ctx.d;
    pop = double(rand(nA, d) > 0.5);
    for i = 1:nA, pop(i, :) = repairMask(pop(i, :)); end
    fit = zeros(1, nA);
    for i = 1:nA, fit(i) = fitness(pop(i, :), ctx); end
    [bestFit, bi] = max(fit); best = pop(bi, :);
end

function best = runRIME(ctx, nA, nIter)
    [pop, fit, best, bestFit] = initPop(ctx, nA);
    d = ctx.d;
    for t = 1:nIter
        rimeFactor = t / nIter;                       % anneal explore->exploit
        for i = 1:nA
            if rand < (1 - rimeFactor)
                child = pop(i, :);
                fl = rand(1, d) < 0.15 * (1 - rimeFactor + 0.1);
                child(fl) = 1 - child(fl);
            else
                cr = rand(1, d) < rimeFactor * 0.5;
                child = pop(i, :); child(cr) = best(cr);
            end
            child = repairMask(child);
            f = fitness(child, ctx);
            if f >= fit(i), pop(i, :) = child; fit(i) = f; end
            if f > bestFit, best = child; bestFit = f; end
        end
    end
end

function best = runPLO(ctx, nA, nIter)
    [pop, fit, best, bestFit] = initPop(ctx, nA);
    d = ctx.d;
    for t = 1:nIter
        for i = 1:nA
            if rand < 0.5
                child = pop(i, :);
                fl = rand(1, d) < 0.08; child(fl) = 1 - child(fl);   % aurora drift
            else
                j = randi(nA);
                cr = rand(1, d) < 0.3;
                child = pop(i, :); child(cr) = pop(j, cr);           % recombination
            end
            child = repairMask(child);
            f = fitness(child, ctx);
            if f >= fit(i), pop(i, :) = child; fit(i) = f; end
            if f > bestFit, best = child; bestFit = f; end
        end
    end
end

function best = runHGS(ctx, nA, nIter)
    [pop, fit, best, bestFit] = initPop(ctx, nA);
    d = ctx.d;
    for t = 1:nIter
        hunger = 1 - (fit / (max(fit) + 1e-9));       % worse agents hungrier
        for i = 1:nA
            if rand < hunger(i) * 0.5
                child = pop(i, :);
                fl = rand(1, d) < 0.15; child(fl) = 1 - child(fl);
            else
                j = randi(nA);
                cr = rand(1, d) < 0.2;
                child = pop(i, :); child(cr) = pop(j, cr);
            end
            child = repairMask(child);
            f = fitness(child, ctx);
            if f >= fit(i), pop(i, :) = child; fit(i) = f; end
            if f > bestFit, best = child; bestFit = f; end
        end
    end
end

function best = runGA(ctx, nA, nIter)
    [pop, fit, best, bestFit] = initPop(ctx, nA);
    d = ctx.d;
    for t = 1:nIter
        newPop = zeros(nA, d);
        for i = 1:nA
            a = randi(nA); b = randi(nA);
            if fit(a) >= fit(b), p1 = pop(a, :); else, p1 = pop(b, :); end
            a = randi(nA); b = randi(nA);
            if fit(a) >= fit(b), p2 = pop(a, :); else, p2 = pop(b, :); end
            cut = randi(d - 1);
            child = [p1(1:cut), p2(cut+1:end)];
            fl = rand(1, d) < (1 / d); child(fl) = 1 - child(fl);
            newPop(i, :) = repairMask(child);
        end
        pop = newPop;
        for i = 1:nA, fit(i) = fitness(pop(i, :), ctx); end
        [mx, mi] = max(fit);
        if mx > bestFit, best = pop(mi, :); bestFit = mx; end
    end
end

function best = runPSO(ctx, nA, nIter)
    d = ctx.d;
    pos = double(rand(nA, d) > 0.5); vel = (rand(nA, d) * 8) - 4;
    fit = zeros(1, nA);
    for i = 1:nA, fit(i) = fitness(repairMask(pos(i, :)), ctx); end
    pbest = pos; pf = fit;
    [bestFit, gi] = max(fit); best = repairMask(pos(gi, :)); gpos = pos(gi, :);
    w = 0.7; c1 = 1.5; c2 = 1.5;
    for t = 1:nIter
        for i = 1:nA
            r1 = rand(1, d); r2 = rand(1, d);
            vel(i, :) = w*vel(i, :) + c1*r1.*(pbest(i, :) - pos(i, :)) + c2*r2.*(gpos - pos(i, :));
            vel(i, :) = max(min(vel(i, :), 4), -4);
            sig = 1 ./ (1 + exp(-vel(i, :)));
            pos(i, :) = double(rand(1, d) < sig);
            f = fitness(repairMask(pos(i, :)), ctx);
            if f > pf(i), pbest(i, :) = pos(i, :); pf(i) = f; end
            if f > bestFit, bestFit = f; best = repairMask(pos(i, :)); gpos = pos(i, :); end
        end
    end
end

% ======================================================================= %
function c = repairMask(c)
    if sum(c) == 0, c(randi(numel(c))) = 1; end
end

function f = fitness(mask, ctx)
    sel = find(mask == 1);
    if isempty(sel), f = 0.0; return; end
    Xs = ctx.X(:, sel); y = ctx.y;
    folds = stratFolds(y, ctx.cvFolds);
    accs = zeros(ctx.cvFolds, 1); aucs = [];
    for k = 1:ctx.cvFolds
        te = (folds == k); tr = ~te;
        mu = mean(Xs(tr, :), 1); sd = std(Xs(tr, :), 0, 1) + 1e-10;
        Xtr = (Xs(tr, :) - mu) ./ sd; Xte = (Xs(te, :) - mu) ./ sd;
        [pred, proba] = knnClassify(Xtr, y(tr), Xte, ctx.kNN);
        accs(k) = mean(pred == y(te));
        if numel(unique(y(te))) > 1, aucs(end+1) = aucScore(y(te), proba); end %#ok<AGROW>
    end
    acc = mean(accs);
    if strcmpi(ctx.metric, 'auc') && ~isempty(aucs)
        skill = mean(aucs);
    elseif strcmpi(ctx.metric, 'balanced') && ~isempty(aucs)
        skill = 0.5 * acc + 0.5 * mean(aucs);
    else
        skill = acc;
    end
    parsimony = 1 - numel(sel) / ctx.d;
    f = ctx.alpha * skill + (1 - ctx.alpha) * parsimony;
end

function [pred, proba] = knnClassify(Xtr, ytr, Xte, k)
    n = size(Xte, 1); pred = zeros(n, 1); proba = zeros(n, 1);
    for i = 1:n
        dd = sum((Xtr - Xte(i, :)).^2, 2);
        [~, ord] = sort(dd, 'ascend');
        nn = ytr(ord(1:min(k, numel(ord))));
        proba(i) = mean(nn); pred(i) = double(proba(i) >= 0.5);
    end
end

function a = aucScore(y, proba)
    pos = proba(y == 1); neg = proba(y == 0);
    if isempty(pos) || isempty(neg), a = 0.5; return; end
    c = 0;
    for i = 1:numel(pos), c = c + sum(pos(i) > neg) + 0.5*sum(pos(i) == neg); end
    a = c / (numel(pos) * numel(neg));
end

function folds = stratFolds(y, k)
    folds = zeros(numel(y), 1);
    for cls = unique(y)'
        idx = find(y == cls);
        idx = idx(randperm(numel(idx)));
        folds(idx) = mod(0:numel(idx)-1, k) + 1;
    end
end
