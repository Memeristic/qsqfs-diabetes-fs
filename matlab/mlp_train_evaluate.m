function metrics = mlp_train_evaluate(Xtr, ytr, Xte, yte, cfg, seed)
% MLP_TRAIN_EVALUATE  Toolbox-free MLP classifier (manual forward/backward),
% portable to plain MATLAB or GNU Octave with NO Deep Learning / Statistics
% Toolbox dependency.
%
%   metrics = mlp_train_evaluate(Xtr, ytr, Xte, yte, cfg, seed)
%
% Mirrors src/neural_model.py's design: early stopping restores the
% best-validation-loss weights at the end. Architecture: 2 ReLU hidden
% layers + sigmoid output, L2-regularised, trained by mini-batch gradient
% descent with momentum. A validation split is carved out of the TRAINING
% fold only (no peeking at the outer-CV test fold).
%
% cfg fields (all optional, defaults mirror config.yaml's neural_model:):
%   hidden_layers (default [64 32]), l2_alpha (1e-4), batch_size (32),
%   max_epochs (200), early_stopping_patience (15), learning_rate (0.001)
%
% OUTPUT metrics: struct with .accuracy, .auc, .f1, .proba, .y_test

    if nargin < 6, seed = 42; end
    if nargin < 5, cfg = struct(); end
    cfg = withDefault(cfg, 'hidden_layers', [64 32]);
    cfg = withDefault(cfg, 'l2_alpha', 1e-4);
    cfg = withDefault(cfg, 'batch_size', 32);
    cfg = withDefault(cfg, 'max_epochs', 200);
    cfg = withDefault(cfg, 'early_stopping_patience', 15);
    cfg = withDefault(cfg, 'learning_rate', 1e-3);
    rng(seed, 'twister');

    % ---- scale on TRAIN fold only, then carve an internal val split ----
    mu = mean(Xtr, 1); sd = std(Xtr, 0, 1) + 1e-10;
    Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;

    n = size(Xtr, 1);
    if n < 10
        metrics = knn_fallback(Xtr, ytr, Xte, yte); return;
    end
    idx = randperm(n);
    nVal = max(5, round(0.15 * n));
    valIdx = idx(1:nVal); trIdx = idx(nVal+1:end);
    Xv = Xtr(valIdx, :); yv = ytr(valIdx);
    Xt = Xtr(trIdx, :);  yt = ytr(trIdx);

    layers = [size(Xt, 2), cfg.hidden_layers(:)', 1];
    W = cell(1, numel(layers)-1); B = cell(1, numel(layers)-1);
    for l = 1:numel(layers)-1
        W{l} = randn(layers(l), layers(l+1)) * sqrt(2 / layers(l));
        B{l} = zeros(1, layers(l+1));
    end

    bestValLoss = inf; bestW = W; bestB = B; patienceLeft = cfg.early_stopping_patience;
    nT = size(Xt, 1); bs = min(cfg.batch_size, nT);
    velW = cellfun(@(w) zeros(size(w)), W, 'UniformOutput', false);
    velB = cellfun(@(b) zeros(size(b)), B, 'UniformOutput', false);
    momentum = 0.9; lr = cfg.learning_rate;

    for epoch = 1:cfg.max_epochs
        perm = randperm(nT);
        for s = 1:bs:nT
            bidx = perm(s:min(s+bs-1, nT));
            xb = Xt(bidx, :); yb = yt(bidx);
            [W, B, velW, velB] = trainStep(W, B, velW, velB, xb, yb, lr, momentum, cfg.l2_alpha);
        end
        [~, valLoss] = forward(W, B, Xv, yv, cfg.l2_alpha);
        if valLoss < bestValLoss - 1e-6
            bestValLoss = valLoss; bestW = W; bestB = B; patienceLeft = cfg.early_stopping_patience;
        else
            patienceLeft = patienceLeft - 1;
            if patienceLeft <= 0, break; end
        end
    end

    % Restore the best-validation weights, not the last epoch.
    W = bestW; B = bestB;
    proba = forwardProba(W, B, Xte);
    pred = double(proba >= 0.5);

    metrics = struct();
    metrics.accuracy = mean(pred == yte);
    metrics.auc = aucScore(yte, proba);
    metrics.f1 = f1Score(yte, pred);
    metrics.proba = proba;
    metrics.y_test = yte;
end

function opts = withDefault(opts, name, val)
    if ~isfield(opts, name) || isempty(opts.(name)), opts.(name) = val; end
end

function [W, B, velW, velB] = trainStep(W, B, velW, velB, x, y, lr, mom, l2)
    L = numel(W);
    A = cell(1, L+1); Z = cell(1, L);
    A{1} = x;
    for l = 1:L
        Z{l} = A{l} * W{l} + B{l};
        if l < L, A{l+1} = max(0, Z{l});          % ReLU
        else,     A{l+1} = 1 ./ (1 + exp(-Z{l})); % sigmoid
        end
    end
    yhat = A{L+1};
    m = size(x, 1);
    dZ = yhat - y(:);                              % dL/dz for sigmoid+BCE
    for l = L:-1:1
        dW = (A{l}' * dZ) / m + l2 * W{l};
        dB = mean(dZ, 1);
        if l > 1
            dA = dZ * W{l}';
            dZ = dA .* (Z{l-1} > 0);               % ReLU derivative
        end
        velW{l} = mom * velW{l} - lr * dW;
        velB{l} = mom * velB{l} - lr * dB;
        W{l} = W{l} + velW{l};
        B{l} = B{l} + velB{l};
    end
end

function [proba, loss] = forward(W, B, x, y, l2)
    proba = forwardProba(W, B, x);
    eps_ = 1e-9;
    bce = -mean(y(:).*log(proba+eps_) + (1-y(:)).*log(1-proba+eps_));
    reg = 0;
    for l = 1:numel(W), reg = reg + l2 * sum(W{l}(:).^2); end
    loss = bce + reg;
end

function proba = forwardProba(W, B, x)
    a = x;
    for l = 1:numel(W)
        z = a * W{l} + B{l};
        if l < numel(W), a = max(0, z); else, a = 1 ./ (1 + exp(-z)); end
    end
    proba = a(:);
end

function metrics = knn_fallback(Xtr, ytr, Xte, yte)
    % Degenerate-size guard (too few training rows for an MLP split).
    n = size(Xte, 1); pred = zeros(n, 1);
    for i = 1:n
        d = sum((Xtr - Xte(i, :)).^2, 2);
        [~, ord] = sort(d, 'ascend');
        k = min(3, numel(ord));
        pred(i) = mode(ytr(ord(1:k)));
    end
    metrics = struct('accuracy', mean(pred == yte), 'auc', aucScore(yte, pred), ...
                     'f1', f1Score(yte, pred), 'proba', pred, 'y_test', yte);
end

function a = aucScore(y, proba)
    y = y(:); proba = proba(:);
    pos = proba(y == 1); neg = proba(y == 0);
    if isempty(pos) || isempty(neg), a = NaN; return; end
    cnt = 0;
    for i = 1:numel(pos)
        cnt = cnt + sum(pos(i) > neg) + 0.5 * sum(pos(i) == neg);
    end
    a = cnt / (numel(pos) * numel(neg));
end

function f = f1Score(y, pred)
    y = y(:); pred = pred(:);
    tp = sum(pred == 1 & y == 1); fp = sum(pred == 1 & y == 0); fn = sum(pred == 0 & y == 1);
    if tp == 0, f = 0; return; end
    prec = tp / (tp + fp); rec = tp / (tp + fn);
    f = 2 * prec * rec / (prec + rec + 1e-12);
end
