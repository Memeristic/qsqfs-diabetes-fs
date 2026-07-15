function results = ablation_study(X, y, cfg, nTrials, seed)
% ABLATION_STUDY  Leak-free ablation of QSQ-FS's four mechanisms.
%
%   results = ablation_study(X, y, cfg, nTrials, seed)
%
% Mirrors src/ablation.py: each variant is evaluated with select_then_score
% (select on an outer TRAIN fold, score on the matching held-out TEST fold),
% so ablation deltas are measured leak-free rather than on a full-data fit.
% Variants use QSQ_FS.m's useQS/useQQ/useCache/elitism switches to disable each
% mechanism independently.
%
% cfg : struct with a single QSQ_FS-shaped param struct, e.g. built by
%       nested_cv_pipeline's qsqfsParams (n_colonies, max_iter, etc.) -- pass
%       cfg.stage2 fields directly (single-stage evaluation, matching the
%       Python ablation harness which runs one combined-stage QSQ-FS per
%       variant rather than the full two-stage funnel).
% nTrials : outer folds (default 5)

    if nargin < 4 || isempty(nTrials), nTrials = 5; end
    if nargin < 5, seed = 42; end
    d = size(X, 2);

    variants = struct( ...
        'Full',        struct(), ...
        'No_QS',       struct('useQS', false), ...
        'No_QQ',       struct('useQQ', false), ...
        'No_Cache',    struct('useCache', false), ...
        'No_Elitism',  struct('elitism', false));

    vnames = fieldnames(variants);
    results = struct();
    folds = stratifiedKFold(y, nTrials, seed);

    for vi = 1:numel(vnames)
        vname = vnames{vi};
        extra = variants.(vname);
        accs = []; nfs = []; tStart = tic;

        for fold = 1:nTrials
            te = (folds == fold); tr = ~te;
            Xtr = X(tr, :); Xte = X(te, :); ytr = y(tr); yte = y(te);

            p = baseParams(cfg, seed + fold);
            fn = fieldnames(extra);
            for k = 1:numel(fn), p.(fn{k}) = extra.(fn{k}); end

            [sel, ~, ~] = QSQ_FS(Xtr, ytr, p);
            if isempty(sel), sel = 1:d; end
            m = knn_score_holdout(Xtr(:, sel), ytr, Xte(:, sel), yte, p.kNN);
            accs(end+1) = m.accuracy; nfs(end+1) = numel(sel); %#ok<AGROW>
        end
        elapsed = toc(tStart);

        r = struct('accuracy', accs, 'n_features', nfs, 'runtime', elapsed);
        results.(vname) = r;
        fprintf('%-12s acc=%.4f+/-%.4f  feat=%.1f  runtime=%.2fs\n', ...
                vname, mean(accs), std(accs), mean(nfs), elapsed);
    end
end

function p = baseParams(cfg, seed)
    p = struct();
    p.nColonies = getf(cfg, 'n_colonies', 30);
    p.maxIter1 = getf(cfg, 'max_iter_stage1', 0);
    p.maxIter2 = getf(cfg, 'max_iter_stage2', 20);
    p.alpha = getf(cfg, 'alpha', 0.95);
    p.wAI = getf(cfg, 'w_AI', 0.50);
    p.delta2 = getf(cfg, 'delta2', 0.95);
    p.rho = getf(cfg, 'rho', 0.80);
    p.kNN = getf(cfg, 'k_nn', 3);
    p.cvFolds = getf(cfg, 'cv_folds', 5);
    p.stagnationWindow = getf(cfg, 'stagnation_window', 15);
    p.diversityThresh = getf(cfg, 'diversity_thresh', 0.05);
    p.seed = seed; p.verbose = false;
end

function v = getf(s, name, default)
    if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end

function folds = stratifiedKFold(y, k, seed)
    rng(seed, 'twister');
    folds = zeros(numel(y), 1);
    for cls = unique(y)'
        idx = find(y == cls);
        idx = idx(randperm(numel(idx)));
        assign = mod(0:numel(idx)-1, k) + 1;
        folds(idx) = assign;
    end
end

function metrics = knn_score_holdout(Xtr, ytr, Xte, yte, k)
    if nargin < 5 || isempty(k), k = 3; end
    mu = mean(Xtr, 1); sd = std(Xtr, 0, 1) + 1e-10;
    Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;
    n = size(Xte, 1); pred = zeros(n, 1);
    for i = 1:n
        dd = sum((Xtr - Xte(i, :)).^2, 2);
        [~, ord] = sort(dd, 'ascend');
        nn = ytr(ord(1:min(k, numel(ord))));
        pred(i) = mode(nn);
    end
    metrics = struct('accuracy', mean(pred == yte));
end
