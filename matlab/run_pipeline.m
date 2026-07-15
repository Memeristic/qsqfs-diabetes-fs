%% run_pipeline.m
% Main MATLAB driver -- mirrors verify_setup.py's end-to-end smoke run.
%
% Usage:
%   >> run_pipeline                          % synthetic demo data
%   >> run_pipeline('real', '/path/to/mimic-iv')   % real MIMIC-IV (explicit path)
%   >> run_pipeline('real')                  % real MIMIC-IV -- pops up a FOLDER PICKER
%   >> run_pipeline('ui')                    % pops up a DATASET CHOOSER dialog:
%                                            %   Demo / MIMIC-IV folder / single CSV
%   >> run_pipeline('csv', '/path/data.csv') % run on an arbitrary tidy CSV
%                                            %   (must have a 'label' column)
%
% The 'ui' and path-less 'real' modes open native file/folder dialogs
% (uigetdir / uigetfile), so you can point-and-click your dataset instead of
% typing a path -- this is the MATLAB counterpart to the Streamlit app's
% data-source selector. If you run MATLAB/Octave with no GUI (e.g. -nodisplay),
% the dialogs can't open; pass the path explicitly instead.
%
% For REAL MIMIC-IV, the chosen folder must be laid out like the official
% download (dataRoot/hosp/*.csv and dataRoot/icu/*.csv, decompressed). This
% script cannot download or credential MIMIC-IV for you. See matlab/README.md.

function run_pipeline(mode, pathArg)
    if nargin < 1, mode = 'demo'; end
    addpath(fileparts(mfilename('fullpath')));
    t0 = tic;

    % ---- Resolve the dataset into X, y, names, mmap ----
    switch lower(mode)
        case 'ui'
            [X, y, names, mmap] = chooseDatasetDialog();
        case 'csv'
            if nargin < 2, error('run_pipeline:missingPath', ...
                    'Usage: run_pipeline(''csv'', ''/path/to/data.csv'')'); end
            [X, y, names, mmap] = load_generic_csv(pathArg);
        case 'real'
            if nargin < 2 || isempty(pathArg)
                pathArg = pickFolderDialog();          % <-- FOLDER PICKER POPUP
                if isempty(pathArg), return; end
            end
            fprintf('=== [1/6] Loading data (real MIMIC-IV) ===\n');
            data = load_mimic_data(pathArg);
            [X, y, names, mmap] = buildFromData(data);
        otherwise   % 'demo'
            fprintf('=== [1/6] Loading data (demo) ===\n');
            data = generate_synthetic_data(1000, 42);
            [X, y, names, mmap] = buildFromData(data);
    end

    runPipelineCore(X, y, names, mmap, t0);
end

% ======================================================================= %
% Shared pipeline core (steps 2-6) -- used by every dataset path          %
% ======================================================================= %
function runPipelineCore(X, y, names, mmap, t0)
    fprintf('\nMatrix: %d patients x %d features, prevalence=%.1f%%\n', ...
            size(X,1), size(X,2), 100*mean(y));

    cfg = defaultCfg();

    fprintf('\n=== [3/6] Nested-CV pipeline (leak-free) ===\n');
    res = nested_cv_pipeline(X, y, mmap, cfg, 5, 'knn', 42);
    fprintf('QSQ-FS nested CV: AUC=%.4f (+/-%.4f)  ACC=%.4f  Features=%.1f\n', ...
            res.auc_mean, res.auc_ci95, res.accuracy_mean, res.n_features_mean);

    fprintf('\n=== [4/6] Comparative analysis (equal budget) ===\n');
    comparative_baselines(X, y, cfg.stage2, 5, 30, 20, 42);

    fprintf('\n=== [5/6] Ablation study (real switches) ===\n');
    ablation_study(X, y, cfg.stage2, 5, 42);

    fprintf('\n=== [6/6] Clinical alignment of stable features ===\n');
    stable = names(res.selection_frequency >= 0.5);
    fprintf('Stable features (selected in >=50%% of folds): %s\n', strjoin(stable, ', '));

    fprintf('\nAll steps completed in %.1fs.\n', toc(t0));
end

% ======================================================================= %
% Helpers                                                                 %
% ======================================================================= %
function [X, y, names, mmap] = buildFromData(data)
    fprintf('\n=== [2/6] Building modalities ===\n');
    opts = struct('combine_how', 'outer', 'top_labs', 50, 'top_vitals', 30, ...
                  'top_meds', 50, 'top_dx', 30);
    [X, y, names, mmap] = build_modalities(data, opts);
end

function cfg = defaultCfg()
    cfg = struct();
    cfg.stage1 = struct('n_colonies', 20, 'max_iter_stage1', 10, 'cv_folds', 5, ...
                        'k_nn', 3, 'alpha', 0.95, 'w_AI', 0.50, 'delta1', 0.97, ...
                        'weak_thresh1', 0.30, 'stagnation_window', 15, 'diversity_thresh', 0.05);
    cfg.stage2 = struct('n_colonies', 30, 'max_iter_stage2', 25, 'cv_folds', 5, ...
                        'k_nn', 3, 'alpha', 0.95, 'w_AI', 0.50, 'delta2', 0.95, ...
                        'rho', 0.80, 'stagnation_window', 15, 'diversity_thresh', 0.05);
    cfg.neural_model = struct('hidden_layers', [64 32], 'l2_alpha', 1e-4, ...
                              'batch_size', 32, 'max_epochs', 150, ...
                              'early_stopping_patience', 12, 'learning_rate', 1e-3);
end

function dataRoot = pickFolderDialog()
% Native folder picker for a MIMIC-IV directory. Returns '' if cancelled or if
% no GUI is available (headless).
    dataRoot = '';
    if exist('uigetdir', 'file') == 0
        fprintf(['No folder-picker available in this environment. Re-run with an ' ...
                 'explicit path:\n  run_pipeline(''real'', ''/path/to/mimic-iv'')\n']);
        return;
    end
    sel = uigetdir(pwd, 'Select the MIMIC-IV folder (the one containing hosp/ and icu/)');
    if isequal(sel, 0)
        fprintf('Folder selection cancelled.\n');
        return;
    end
    dataRoot = sel;
    fprintf('Selected folder: %s\n', dataRoot);
end

function [X, y, names, mmap] = chooseDatasetDialog()
% Pop up a small chooser: Demo / MIMIC-IV folder / single CSV, then the right
% picker. This is the MATLAB analogue of the Streamlit "Data source" selector.
    choice = '';
    if exist('questdlg', 'file')
        choice = questdlg('Which dataset would you like to run QSQ-FS on?', ...
                          'QSQ-FS -- choose a dataset', ...
                          'Demo (synthetic)', 'MIMIC-IV folder...', 'Single CSV file...', ...
                          'Demo (synthetic)');
    else
        % text-menu fallback (no GUI)
        fprintf('Choose a dataset:\n  1) Demo (synthetic)\n  2) MIMIC-IV folder\n  3) Single CSV file\n');
        k = input('Enter 1, 2 or 3: ');
        opts = {'Demo (synthetic)', 'MIMIC-IV folder...', 'Single CSV file...'};
        if isempty(k) || k < 1 || k > 3, k = 1; end
        choice = opts{k};
    end

    switch choice
        case 'MIMIC-IV folder...'
            root = pickFolderDialog();
            if isempty(root), error('run_pipeline:cancelled', 'No folder selected.'); end
            fprintf('=== [1/6] Loading data (real MIMIC-IV) ===\n');
            data = load_mimic_data(root);
            [X, y, names, mmap] = buildFromData(data);
        case 'Single CSV file...'
            if exist('uigetfile', 'file') == 0
                error('run_pipeline:noGui', ...
                      'No file-picker here; use run_pipeline(''csv'', ''/path/data.csv'').');
            end
            [f, p] = uigetfile({'*.csv', 'CSV files (*.csv)'}, 'Select a CSV dataset');
            if isequal(f, 0), error('run_pipeline:cancelled', 'No file selected.'); end
            fprintf('=== [1/6] Loading data (generic CSV) ===\n');
            [X, y, names, mmap] = load_generic_csv(fullfile(p, f));
        otherwise   % Demo
            fprintf('=== [1/6] Loading data (demo) ===\n');
            data = generate_synthetic_data(1000, 42);
            [X, y, names, mmap] = buildFromData(data);
    end
end
