% demo_QSQ_FS.m  -- minimal example of the QSQ_FS reference implementation.
rng(42);
n = 300; d = 20;
X = randn(n, d);
w = zeros(1, d); w([3 7 11]) = [1.4 -1.1 0.9];      % only a few informative features
y = double((X * w' + 0.5 * randn(n, 1)) > 0);

params = struct('nColonies', 20, 'maxIter1', 12, 'maxIter2', 15, ...
                'cvFolds', 5, 'seed', 42, 'verbose', true);
[selected, bestFitness, history] = QSQ_FS(X, y, params);

fprintf('\nSelected %d features: %s\n', numel(selected), mat2str(selected));
fprintf('Best fitness: %.4f\n', bestFitness);

figure;
plot(history.fitness, 'LineWidth', 2); hold on;
xline(history.stage1Boundary, '--');
xlabel('Iteration'); ylabel('Best fitness'); title('QSQ-FS convergence (MATLAB)');
