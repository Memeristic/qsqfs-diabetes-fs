% demo_fusion_and_optimizers.m
% -----------------------------------------------------------------------------
% Minimal demonstration of the two proposal-driven additions:
%   (1) the named metaheuristics RIME / PLO / HGS as first-class feature
%       selectors (feature_selectors.m), and
%   (2) multimodal feature-/decision-/hybrid-level fusion (multimodal_fusion.m).
% Runs on synthetic data so it needs no MIMIC-IV download; portable to Octave.

rng(42);
n = 300; d = 24;
X = randn(n, d);
w = zeros(1, d); w([2 5 9 14 19]) = [1.3 -1.0 0.9 1.1 -0.8];   % informative feats
y = double((X * w' + 0.6 * randn(n, 1)) > 0);

% two synthetic "modalities": columns 1-12 and 13-24
modalityMap = struct('modA', 1:12, 'modB', 13:24);

% ---- 80/20 leak-free split ----
cv = false(n, 1); cv(1:round(0.2*n)) = true; cv = cv(randperm(n));
Xtr = X(~cv, :); ytr = y(~cv); Xte = X(cv, :); yte = y(cv);

% ---- (1) named optimisers under an equal budget ----
fprintf('\n=== Named metaheuristic feature selectors (equal budget) ===\n');
for m = {'RIME', 'PLO', 'HGS', 'GA', 'PSO'}
    sel = feature_selectors(m{1}, Xtr, ytr, 20, 12, 42, 3, 3, 'balanced');
    fprintf('  %-5s selected %2d features: %s\n', m{1}, numel(sel), mat2str(sel));
end

% ---- (2) fusion strategies on the same split ----
fprintf('\n=== Multimodal fusion strategies ===\n');
cfg = struct('hidden_layers', [32], 'l2_alpha', 1e-3, 'latent_dim', 8);
for s = {'feature', 'decision', 'hybrid'}
    m = multimodal_fusion(Xtr, ytr, Xte, yte, modalityMap, s{1}, cfg, 42);
    fprintf('  %-8s  AUC=%.3f  ACC=%.3f  F1=%.3f\n', s{1}, m.auc, m.accuracy, m.f1);
end

fprintf('\nDone. (hybrid should be >= the weaker of feature/decision.)\n');
