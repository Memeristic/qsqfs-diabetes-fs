function metrics = multimodal_fusion(Xtr, ytr, Xte, yte, modalityMap, strategy, cfg, seed)
% MULTIMODAL_FUSION  Feature-, decision-, or hybrid-level multimodal fusion,
% the MATLAB counterpart of src/fusion.py.
%
%   metrics = multimodal_fusion(Xtr, ytr, Xte, yte, modalityMap, strategy, cfg, seed)
%
% STRATEGIES (thesis Sec 2.2 / 2.5, "hybrid fusion")
%   'feature'  : each modality is encoded into a small latent code by its own
%                MLP; the latent codes are concatenated into a SHARED latent
%                space and one head is trained on the joint representation.
%   'decision' : one classifier per modality; per-modality probabilities are
%                combined by a validation-AUC weighting (ensemble).
%   'hybrid'   : average of the feature-level and decision-level probabilities.
%
% Toolbox-free: every encoder / head is the manual-backprop MLP from
% mlp_train_evaluate.m, so this runs on base MATLAB and GNU Octave with no
% Deep Learning / Statistics Toolbox. The minority class is oversampled in the
% joint head so an imbalanced (diabetic) minority is not swamped -- the same
% class-weighting idea as src/fusion.py.
%
% INPUTS
%   Xtr/ytr/Xte/yte : train/test split (leak-free: caller keeps test unseen)
%   modalityMap     : struct mapping modality name -> LOCAL column indices into
%                     the columns of Xtr/Xte (i.e. already restricted to the
%                     selected feature subset)
%   strategy        : 'feature' | 'decision' | 'hybrid' (default 'hybrid')
%   cfg             : neural-model cfg (hidden_layers, l2_alpha, ...); optional
%   seed            : RNG seed (default 42)
%
% OUTPUT metrics : struct with .accuracy, .auc, .f1, .proba, .y_test

    if nargin < 6 || isempty(strategy), strategy = 'hybrid'; end
    if nargin < 7, cfg = struct(); end
    if nargin < 8, seed = 42; end
    rng(seed, 'twister');

    names = fieldnames(modalityMap);
    if isempty(names)
        modalityMap = struct('all', 1:size(Xtr, 2)); names = {'all'};
    end
    latentDim = getf(cfg, 'latent_dim', 8);

    % ---- decision-level: one classifier per modality + AUC weights ----
    pFeatTe = zeros(size(Xte, 1), 1);          % accumulate feature-level later
    modProbaTe = zeros(size(Xte, 1), numel(names));
    modW = zeros(1, numel(names));
    for i = 1:numel(names)
        cols = modalityMap.(names{i});
        m = trainHead(Xtr(:, cols), ytr, cfg, seed + i);
        [ptr, ~] = headProba(m, Xtr(:, cols));    % train proba for AUC weight
        modProbaTe(:, i) = headProba(m, Xte(:, cols));
        modW(i) = max(aucScore(ytr, ptr) - 0.5, 1e-3);
    end
    if sum(modW) == 0, modW = ones(1, numel(names)); end
    pDecTe = (modProbaTe * modW(:)) / sum(modW);

    % ---- feature-level: per-modality encoders -> shared latent -> joint head ----
    Ztr = []; Zte = [];
    for i = 1:numel(names)
        cols = modalityMap.(names{i});
        enc = trainEncoder(Xtr(:, cols), ytr, latentDim, cfg, seed + 100 + i);
        Ztr = [Ztr, encodeLatent(enc, Xtr(:, cols))]; %#ok<AGROW>
        Zte = [Zte, encodeLatent(enc, Xte(:, cols))]; %#ok<AGROW>
    end
    [Ztr_b, ytr_b] = oversampleMinority(Ztr, ytr, seed);
    joint = trainHead(Ztr_b, ytr_b, cfg, seed + 7);
    pFeatTe = headProba(joint, Zte);

    % ---- combine ----
    switch lower(strategy)
        case 'feature',  proba = pFeatTe;
        case 'decision', proba = pDecTe;
        otherwise,       proba = 0.5 * (pFeatTe + pDecTe);
    end

    pred = double(proba >= 0.5);
    metrics = struct('accuracy', mean(pred == yte), 'auc', aucScore(yte, proba), ...
                     'f1', f1Score(yte, pred), 'proba', proba, 'y_test', yte);
end

% ======================================================================= %
function m = trainHead(X, y, cfg, seed)
    % thin wrapper around the manual-backprop MLP; returns fitted weights
    hidden = getf(cfg, 'hidden_layers', [32]);
    l2 = getf(cfg, 'l2_alpha', 1e-3);
    m = mlpFit(X, y, hidden, l2, seed);
end

function [proba, pred] = headProba(m, X)
    proba = mlpProba(m, X); pred = double(proba >= 0.5);
end

function enc = trainEncoder(X, y, latentDim, cfg, seed)
    l2 = getf(cfg, 'l2_alpha', 1e-3);
    enc = mlpFit(X, y, max(2, latentDim), l2, seed);   % single bottleneck layer
end

function Z = encodeLatent(enc, X)
    % forward pass up to (not including) the output layer -> latent code
    mu = enc.mu; sd = enc.sd;
    a = (X - mu) ./ sd;
    for l = 1:numel(enc.W) - 1
        a = max(0, a * enc.W{l} + enc.B{l});
    end
    Z = a;
end

function [Xb, yb] = oversampleMinority(X, y, seed)
    rng(seed + 999, 'twister');
    n1 = sum(y == 1); n0 = sum(y == 0);
    if n1 == 0 || n0 == 0 || n1 == n0, Xb = X; yb = y; return; end
    if n1 < n0, minC = 1; need = n0 - n1; else, minC = 0; need = n1 - n0; end
    idx = find(y == minC);
    extra = idx(randi(numel(idx), need, 1));
    Xb = [X; X(extra, :)]; yb = [y; y(extra)];
end

% ---- minimal manual MLP (scale -> ReLU hidden -> sigmoid), shared by heads ----
function m = mlpFit(X, y, hidden, l2, seed)
    rng(seed, 'twister');
    mu = mean(X, 1); sd = std(X, 0, 1) + 1e-10;
    Xs = (X - mu) ./ sd;
    layers = [size(Xs, 2), hidden(:)', 1];
    W = cell(1, numel(layers) - 1); B = cell(1, numel(layers) - 1);
    for l = 1:numel(layers) - 1
        W{l} = randn(layers(l), layers(l+1)) * sqrt(2 / layers(l));
        B{l} = zeros(1, layers(l+1));
    end
    lr = 0.01; mom = 0.9;
    vW = cellfun(@(w) zeros(size(w)), W, 'UniformOutput', false);
    vB = cellfun(@(b) zeros(size(b)), B, 'UniformOutput', false);
    n = size(Xs, 1); bs = min(32, n);
    for epoch = 1:150
        perm = randperm(n);
        for s = 1:bs:n
            bidx = perm(s:min(s+bs-1, n));
            [W, B, vW, vB] = step(W, B, vW, vB, Xs(bidx, :), y(bidx), lr, mom, l2);
        end
    end
    m = struct('W', {W}, 'B', {B}, 'mu', mu, 'sd', sd);
end

function proba = mlpProba(m, X)
    a = (X - m.mu) ./ m.sd;
    for l = 1:numel(m.W)
        z = a * m.W{l} + m.B{l};
        if l < numel(m.W), a = max(0, z); else, a = 1 ./ (1 + exp(-z)); end
    end
    proba = a(:);
end

function [W, B, vW, vB] = step(W, B, vW, vB, x, y, lr, mom, l2)
    L = numel(W); A = cell(1, L+1); Z = cell(1, L); A{1} = x;
    for l = 1:L
        Z{l} = A{l} * W{l} + B{l};
        if l < L, A{l+1} = max(0, Z{l}); else, A{l+1} = 1 ./ (1 + exp(-Z{l})); end
    end
    m = size(x, 1); dZ = A{L+1} - y(:);
    for l = L:-1:1
        dW = (A{l}' * dZ) / m + l2 * W{l};
        dB = mean(dZ, 1);
        if l > 1, dZ = (dZ * W{l}') .* (Z{l-1} > 0); end
        vW{l} = mom * vW{l} - lr * dW; vB{l} = mom * vB{l} - lr * dB;
        W{l} = W{l} + vW{l}; B{l} = B{l} + vB{l};
    end
end

function a = aucScore(y, proba)
    y = y(:); proba = proba(:);
    pos = proba(y == 1); neg = proba(y == 0);
    if isempty(pos) || isempty(neg), a = 0.5; return; end
    c = 0;
    for i = 1:numel(pos), c = c + sum(pos(i) > neg) + 0.5*sum(pos(i) == neg); end
    a = c / (numel(pos) * numel(neg));
end

function f = f1Score(y, pred)
    y = y(:); pred = pred(:);
    tp = sum(pred==1 & y==1); fp = sum(pred==1 & y==0); fn = sum(pred==0 & y==1);
    if tp == 0, f = 0; return; end
    prec = tp/(tp+fp); rec = tp/(tp+fn);
    f = 2*prec*rec/(prec+rec+1e-12);
end

function v = getf(s, name, default)
    if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end
