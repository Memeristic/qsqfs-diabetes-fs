function results = comparative_baselines(X, y, cfg, nTrials, nAgents, maxIter, seed)
% COMPARATIVE_BASELINES  QSQ-FS vs GA / PSO (equal budget) vs a full-feature
% logistic-regression baseline, all under the same leak-free select-then-score
% protocol as src/comparative_analysis.py.
%
%   results = comparative_baselines(X, y, cfg, nTrials, nAgents, maxIter, seed)
%
% Every metaheuristic selects on the outer TRAIN fold only and is scored with
% KNN on the held-out TEST fold, with an EQUAL population/iteration budget
% (nAgents/maxIter) across QSQ-FS, GA and PSO. An equal-budget comparison is
% the fair one: unequal budgets would bias the comparison toward the method
% given more evaluations.
%
% NOTE ON MATLAB TOOLBOXES: real MATLAB users with the Statistics and Machine
% Learning Toolbox can trivially add fitcsvm/TreeBagger baselines here (they
% are commented out below with drop-in call sites); this file ships toolbox-
% free by default so it also runs unmodified in GNU Octave / base MATLAB.

    if nargin < 4 || isempty(nTrials), nTrials = 5; end
    if nargin < 5 || isempty(nAgents), nAgents = 30; end
    if nargin < 6 || isempty(maxIter), maxIter = 20; end
    if nargin < 7, seed = 42; end

    d = size(X, 2);
    folds = stratifiedKFold(y, nTrials, seed);
    methods = {'QSQ-FS', 'RIME', 'PLO', 'HGS', 'GA', 'PSO', 'LogReg_AllFeatures'};
    results = struct();
    for mi = 1:numel(methods), results.(safeName(methods{mi})) = struct('accuracy', [], 'n_features', []); end

    for fold = 1:nTrials
        te = (folds == fold); tr = ~te;
        Xtr = X(tr, :); Xte = X(te, :); ytr = y(tr); yte = y(te);

        % ---- QSQ-FS (equal budget) ----
        p = struct('nColonies', nAgents, 'maxIter1', 0, 'maxIter2', maxIter, ...
                   'kNN', getf(cfg, 'k_nn', 3), 'cvFolds', getf(cfg, 'cv_folds', 5), ...
                   'seed', seed + fold, 'verbose', false);
        [sel, ~, ~] = QSQ_FS(Xtr, ytr, p);
        if isempty(sel), sel = 1:d; end
        results = scoreInto(results, 'QSQ-FS', Xtr, ytr, Xte, yte, sel);

        % ---- Proposal's named metaheuristics: RIME, PLO, HGS (equal budget) ----
        % plus GA / PSO, all via feature_selectors.m under the same protocol.
        for m = {'RIME', 'PLO', 'HGS', 'GA', 'PSO'}
            name = m{1};
            sel = feature_selectors(name, Xtr, ytr, nAgents, maxIter, seed + fold, ...
                                    p.kNN, p.cvFolds, 'balanced');
            if isempty(sel), sel = 1:d; end
            results = scoreInto(results, name, Xtr, ytr, Xte, yte, sel);
        end

        % ---- Classical full-feature baseline: L2 logistic regression ----
        results = scoreInto(results, 'LogReg_AllFeatures', Xtr, ytr, Xte, yte, 1:d);

        fprintf('Comparative fold %d/%d done.\n', fold, nTrials);
    end

    for mi = 1:numel(methods)
        r = results.(safeName(methods{mi}));
        fprintf('%-20s acc=%.4f+/-%.4f  feat=%.1f\n', methods{mi}, ...
                mean(r.accuracy), std(r.accuracy), mean(r.n_features));
    end
end

% ============================================================ %
function results = scoreInto(results, name, Xtr, ytr, Xte, yte, sel)
    if strcmp(name, 'LogReg_AllFeatures')
        m = logreg_score_holdout(Xtr, ytr, Xte, yte);
    else
        m = knn_score_holdout(Xtr(:, sel), ytr, Xte(:, sel), yte, 3);
    end
    key = safeName(name);
    results.(key).accuracy(end+1) = m.accuracy;
    results.(key).n_features(end+1) = numel(sel);
end

function s = safeName(n)
    s = strrep(strrep(n, '-', '_'), ' ', '_');
end

% ============================================================ %
function metrics = knn_score_holdout(Xtr, ytr, Xte, yte, k)
    mu = mean(Xtr, 1); sd = std(Xtr, 0, 1) + 1e-10;
    Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;
    pred = knnPredict(Xtr, ytr, Xte, k);
    metrics = struct('accuracy', mean(pred == yte));
end

function metrics = logreg_score_holdout(Xtr, ytr, Xte, yte)
    % Toolbox-free L2 logistic regression (Newton-Raphson / IRLS), used as
    % the "strong full-feature classical baseline" (mirrors SVM/RF's role in
    % the Python comparison: a method with NO feature selection at all).
    mu = mean(Xtr, 1); sd = std(Xtr, 0, 1) + 1e-10;
    Xtr = [(Xtr - mu) ./ sd, ones(size(Xtr,1),1)];
    Xte = [(Xte - mu) ./ sd, ones(size(Xte,1),1)];
    d = size(Xtr, 2);
    beta = zeros(d, 1); lambda = 1e-2;
    for it = 1:25
        z = Xtr * beta; p = 1 ./ (1 + exp(-z));
        grad = Xtr' * (p - ytr(:)) / size(Xtr,1) + lambda * beta;
        W = p .* (1 - p);
        H = (Xtr' * (Xtr .* W)) / size(Xtr,1) + lambda * eye(d);
        beta = beta - H \ grad;
    end
    proba = 1 ./ (1 + exp(-Xte * beta));
    pred = double(proba >= 0.5);
    metrics = struct('accuracy', mean(pred == yte));
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

function v = getf(s, name, default)
    if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end
