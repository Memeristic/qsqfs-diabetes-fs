function results = nested_cv_pipeline(X, y, modalityMap, cfg, nOuter, classifier, seed)
% NESTED_CV_PIPELINE  Two-stage QSQ-FS pipeline under outer CV.
%
%   results = nested_cv_pipeline(X, y, modalityMap, cfg, nOuter, classifier, seed)
%
% MATLAB port of src/evaluation.py's nested_cv_evaluate: feature selection
% (Stage 1 per-modality + Stage 2 fusion) runs INSIDE each outer training
% fold only; the outer test fold is never touched until final scoring, which
% is what keeps the reported metrics leakage-free.
%
% INPUTS
%   X, y         : combined feature matrix / binary labels
%   modalityMap  : struct (modality name -> column indices), from build_modalities.m
%   cfg          : struct with .stage1, .stage2, .neural_model sub-structs
%                  (same shape as config.yaml)
%   nOuter       : number of outer folds (default 5)
%   classifier   : 'mlp' | 'knn' (default 'mlp')
%   seed         : base random seed (default 42)
%
% OUTPUT results: struct with per-fold metrics, aggregate mean/std/95%CI,
% and .selection_frequency (fraction of folds each feature was chosen).

    if nargin < 5 || isempty(nOuter), nOuter = 5; end
    if nargin < 6 || isempty(classifier), classifier = 'mlp'; end
    if nargin < 7, seed = 42; end

    [n, d] = size(X);
    folds = stratifiedKFold(y, nOuter, seed);
    selCounts = zeros(1, d);
    accs = []; aucs = []; f1s = []; nfs = [];
    foldSelected = cell(1, nOuter);
    oofProba = []; oofY = [];

    modNames = fieldnames(modalityMap);

    for fold = 1:nOuter
        te = (folds == fold); tr = ~te;
        Xtr = X(tr, :); Xte = X(te, :); ytr = y(tr); yte = y(te);

        % ---- Stage 1: per modality, TRAIN ONLY ----
        pool = []; poolWeight = containers.Map('KeyType', 'double', 'ValueType', 'double');
        for i = 1:numel(modNames)
            cols = modalityMap.(modNames{i});
            if isempty(cols), continue; end
            p1 = qsqfsParams(cfg.stage1, seed + fold, false);
            [selLocal, ~, ~] = QSQ_FS(Xtr(:, cols), ytr, p1);
            for k = 1:numel(selLocal)
                g = cols(selLocal(k));
                pool(end+1) = g; %#ok<AGROW>
                if isKey(poolWeight, g), poolWeight(g) = poolWeight(g) + 1;
                else, poolWeight(g) = 1; end
            end
        end
        pool = unique(pool);
        if isempty(pool), pool = 1:d; end

        % ---- Stage 2: fusion on the Stage-1 candidate pool, TRAIN ONLY ----
        % The Stage-1 candidate pool and its cross-modality selection counts are
        % passed to QSQ_FS as featurePool/poolWeights and the search runs on the
        % full matrix, matching the frequency-weighted Stage-2 seeding in the
        % Python engine (src/qsfs.py fit(..., feature_pool=, pool_weights=)).
        % Returned indices are global, so no remapping is required.
        poolW = zeros(1, numel(pool));
        for k = 1:numel(pool)
            if isKey(poolWeight, pool(k)), poolW(k) = poolWeight(pool(k)); else, poolW(k) = 1; end
        end
        p2 = qsqfsParams(cfg.stage2, seed + fold, true);
        p2.featurePool = pool;
        p2.poolWeights = poolW;
        [selSubIdx, ~, ~] = QSQ_FS(Xtr, ytr, p2);
        if isempty(selSubIdx)
            selected = pool;
        else
            selected = selSubIdx;
        end
        foldSelected{fold} = selected;
        selCounts(selected) = selCounts(selected) + 1;

        % ---- Final classifier on TRAIN, evaluate on TEST ----
        if strcmpi(classifier, 'mlp')
            m = mlp_train_evaluate(Xtr(:, selected), ytr, Xte(:, selected), yte, ...
                                   cfg.neural_model, seed + fold);
        else
            m = knn_score_holdout(Xtr(:, selected), ytr, Xte(:, selected), yte, ...
                                  cfg.stage2.k_nn);
        end
        accs(end+1) = m.accuracy; aucs(end+1) = m.auc; f1s(end+1) = m.f1; %#ok<AGROW>
        nfs(end+1) = numel(selected); %#ok<AGROW>
        oofProba = [oofProba; m.proba(:)]; oofY = [oofY; m.y_test(:)]; %#ok<AGROW>

        fprintf('[nested-CV] fold %d/%d: AUC=%.4f ACC=%.4f F1=%.4f (%d feats)\n', ...
                fold, nOuter, m.auc, m.accuracy, m.f1, numel(selected));
    end

    results = struct();
    results.per_fold = struct('accuracy', accs, 'auc', aucs, 'f1', f1s, 'n_features', nfs);
    results.fold_selected = foldSelected;
    [results.accuracy_mean, results.accuracy_std, results.accuracy_ci95] = ci95(accs);
    [results.auc_mean, results.auc_std, results.auc_ci95] = ci95(aucs(~isnan(aucs)));
    [results.f1_mean, results.f1_std, results.f1_ci95] = ci95(f1s);
    results.n_features_mean = mean(nfs); results.n_features_std = std(nfs);
    results.selection_frequency = selCounts / nOuter;
    results.oof_proba = oofProba; results.oof_y = oofY;
    results.n_outer = nOuter;
end

% ============================================================ %
function p = qsqfsParams(stageCfg, seed, isStage2)
    p = struct();
    p.nColonies = getf(stageCfg, 'n_colonies', 25);
    if isStage2
        p.maxIter1 = 0; p.maxIter2 = getf(stageCfg, 'max_iter_stage2', 40);
        p.delta2 = getf(stageCfg, 'delta2', 0.95); p.rho = getf(stageCfg, 'rho', 0.80);
    else
        p.maxIter1 = getf(stageCfg, 'max_iter_stage1', 15); p.maxIter2 = 0;
        p.delta1 = getf(stageCfg, 'delta1', 0.97);
        p.weakThresh1 = getf(stageCfg, 'weak_thresh1', 0.30);
    end
    p.alpha = getf(stageCfg, 'alpha', 0.95);
    p.wAI = getf(stageCfg, 'w_AI', 0.50);
    p.stagnationWindow = getf(stageCfg, 'stagnation_window', 15);
    p.diversityThresh = getf(stageCfg, 'diversity_thresh', 0.05);
    p.kNN = getf(stageCfg, 'k_nn', 3);
    p.cvFolds = getf(stageCfg, 'cv_folds', 5);
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

function [m, s, ci] = ci95(v)
    v = v(~isnan(v));
    if isempty(v), m = NaN; s = 0; ci = 0; return; end
    m = mean(v); s = std(v);
    ci = 1.96 * s / sqrt(max(1, numel(v)));
end

function metrics = knn_score_holdout(Xtr, ytr, Xte, yte, k)
    if nargin < 5 || isempty(k), k = 3; end
    mu = mean(Xtr, 1); sd = std(Xtr, 0, 1) + 1e-10;
    Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;
    n = size(Xte, 1); proba = zeros(n, 1);
    for i = 1:n
        d = sum((Xtr - Xte(i, :)).^2, 2);
        [~, ord] = sort(d, 'ascend');
        nn = ytr(ord(1:min(k, numel(ord))));
        proba(i) = mean(nn);
    end
    pred = double(proba >= 0.5);
    metrics = struct('accuracy', mean(pred == yte), ...
                     'auc', localAuc(yte, proba), ...
                     'f1', localF1(yte, pred), ...
                     'proba', proba, 'y_test', yte);
end

function a = localAuc(y, proba)
    y = y(:); proba = proba(:);
    pos = proba(y == 1); neg = proba(y == 0);
    if isempty(pos) || isempty(neg), a = NaN; return; end
    cnt = 0;
    for i = 1:numel(pos), cnt = cnt + sum(pos(i) > neg) + 0.5*sum(pos(i) == neg); end
    a = cnt / (numel(pos) * numel(neg));
end

function f = localF1(y, pred)
    y = y(:); pred = pred(:);
    tp = sum(pred==1 & y==1); fp = sum(pred==1 & y==0); fn = sum(pred==0 & y==1);
    if tp == 0, f = 0; return; end
    prec = tp/(tp+fp); rec = tp/(tp+fn);
    f = 2*prec*rec/(prec+rec+1e-12);
end
