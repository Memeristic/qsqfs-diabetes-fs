function [X, y, featureNames, modalityMap, subjectIds] = build_modalities(data, opts)
% BUILD_MODALITIES  Build the combined multimodal feature matrix.
%
%   [X, y, featureNames, modalityMap, subjectIds] = build_modalities(data, opts)
%
% Mirrors src/modality_builder.py, including the real-data fixes documented
% there:
%   * Diabetes-drug leakage filter is a case-insensitive SUBSTRING match
%     against a curated multi-class pattern list (not exact-string equality),
%     so it actually fires on real free-text drug names ("Insulin Glargine",
%     "MetFORMIN (Glucophage) 500 MG").
%   * Diabetes-defining ICD codes (250.*, E08-E13) are stripped from the
%     diagnosis modality (label leakage).
%   * Long-format labs/vitals (real MIMIC-IV: subject_id/itemid/valuenum) are
%     pivoted like labs, not incorrectly averaged over itemid.
%   * Sparse per-patient coverage across pivoted item columns is handled by
%     dropping near-empty columns and mean-imputing the remainder (NOT a hard
%     dropna, which silently zeroes out real, sparse EHR panels).
%
% INPUTS
%   data : struct produced by generate_synthetic_data.m (wide, in-memory) OR
%          load_mimic_data.m (real MIMIC-IV: long-format event tables).
%          Must contain fields: .labs, .vitals, .meds, .diagnoses, .d_icd
%   opts : struct with optional fields:
%          .combine_how  'inner' (default) | 'outer' (mean-impute)
%          .top_labs (50), .top_vitals (30), .top_meds (50), .top_dx (30)
%          .min_coverage (0.05) -- min fraction of patients an item column
%          must cover to be kept (real-data sparsity guard)
%
% OUTPUTS
%   X            : n-by-d combined feature matrix
%   y            : n-by-1 binary label
%   featureNames : 1-by-d cell array of feature names ("labs_glucose",
%                  "labs_item_50931", "meds_Insulin_Glargine", "dx_I10", ...)
%   modalityMap  : struct mapping modality name -> column indices into X
%   subjectIds   : n-by-1 subject_id for the surviving cohort

    if nargin < 1 || ~isstruct(data)
        error('build_modalities:badData', ...
              'data must be the struct returned by load_mimic_data or generate_synthetic_data.');
    end
    if ~isfield(data, 'diagnoses') || isempty(data.diagnoses)
        error('build_modalities:noDiagnoses', ...
              ['data.diagnoses is required -- the diabetes label is derived from ICD codes. ' ...
               'Check that hosp/diagnoses_icd.csv loaded correctly.']);
    end
    if nargin < 2, opts = struct(); end
    opts = withDefault(opts, 'combine_how', 'inner');
    opts = withDefault(opts, 'top_labs', 50);
    opts = withDefault(opts, 'top_vitals', 30);
    opts = withDefault(opts, 'top_meds', 50);
    opts = withDefault(opts, 'top_dx', 30);
    opts = withDefault(opts, 'min_coverage', 0.05);

    label = extractLabel(data);

    mods = struct();
    if isfield(data, 'labs') && ~isempty(data.labs)
        mods.labs = buildNumericModality(data.labs, 'labs', opts.top_labs, opts.min_coverage);
    end
    if isfield(data, 'vitals') && ~isempty(data.vitals)
        mods.vitals = buildNumericModality(data.vitals, 'vitals', opts.top_vitals, opts.min_coverage);
    end
    if isfield(data, 'meds') && ~isempty(data.meds)
        mods.meds = buildMedications(data.meds, opts.top_meds);
    end
    if isfield(data, 'diagnoses') && ~isempty(data.diagnoses)
        mods.dx = buildDiagnoses(data.diagnoses, opts.top_dx);
    end

    [X, y, featureNames, modalityMap, subjectIds] = combineModalities(mods, label, opts.combine_how);

    % --- Post-assembly sanity checks (fail early with an actionable message) ---
    if isempty(X) || size(X, 1) == 0
        error('build_modalities:emptyCohort', ...
              ['No patients survived modality assembly. With combine_how=''inner'' this happens ' ...
               'when no single patient appears in every modality; try opts.combine_how=''outer'' ' ...
               '(mean-imputes missing modalities) or check that the tables share subject_ids.']);
    end
    if size(X, 2) == 0
        error('build_modalities:noFeatures', ...
              'No feature columns were built. Check top_labs/top_vitals/top_meds/top_dx and min_coverage.');
    end
    uy = unique(y);
    if numel(uy) < 2
        error('build_modalities:oneClass', ...
              ['The label has only one class (all %d patients are %s). A classifier needs both ' ...
               'positive and negative cases. This usually means the cohort is too small or the ' ...
               'diagnosis codes were not read correctly.'], numel(y), ...
              ternary(uy(1) == 1, 'diabetic', 'non-diabetic'));
    end
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end

% ============================================================ %
function opts = withDefault(opts, name, val)
    if ~isfield(opts, name), opts.(name) = val; end
end

% ============================================================ %
function label = extractLabel(data)
    % label(subject_id) = 1 iff ANY diagnosis row for that patient is a
    % diabetes-defining ICD code (prefix 250, or E08-E13).
    subs = unique(data.diagnoses.subject_id);
    isDiab = false(size(subs));
    for i = 1:numel(subs)
        rows = data.diagnoses.subject_id == subs(i);
        codes = data.diagnoses.icd_code(rows);
        for k = 1:numel(codes)
            if isDiabetesCode(codes{k}), isDiab(i) = true; break; end
        end
    end
    label = struct('subject_id', subs, 'y', double(isDiab));
    fprintf('Label: %d/%d positive (%.1f%%)\n', sum(isDiab), numel(subs), 100*mean(isDiab));
end

function b = isDiabetesCode(code)
    s = strtrim(char(code));
    b = startsWithAny(s, {'250'}) || startsWithAny(s, {'E08','E09','E10','E11','E12','E13'});
end

function b = startsWithAny(s, prefixes)
    b = false;
    for i = 1:numel(prefixes)
        p = prefixes{i};
        if numel(s) >= numel(p) && strcmpi(s(1:numel(p)), p), b = true; return; end
    end
end

% ============================================================ %
function b = isDiabetesDrugName(txt)
    % Robust real-data drug-leakage filter: case-insensitive substring match
    % against generics + common brand names across every diabetes-drug
    % class. Mirrors src/modality_builder.py's DIABETES_DRUG_RE exactly.
    patterns = {'metformin','glucophage','insulin','lantus','humalog','novolog', ...
        'novorapid','levemir','tresiba','apidra','toujeo','basaglar','humulin','fiasp', ...
        'glipizide','glucotrol','glyburide','glibenclamide','glimepiride','amaryl', ...
        'gliclazide','tolbutamide','chlorpropamide','repaglinide','nateglinide', ...
        'prandin','starlix','pioglitazone','actos','rosiglitazone','avandia', ...
        'acarbose','miglitol','sitagliptin','januvia','saxagliptin','onglyza', ...
        'linagliptin','tradjenta','alogliptin','nesina','liraglutide','victoza', ...
        'exenatide','byetta','bydureon','dulaglutide','trulicity','semaglutide', ...
        'ozempic','rybelsus','lixisenatide','tirzepatide','mounjaro','empagliflozin', ...
        'jardiance','canagliflozin','invokana','dapagliflozin','farxiga', ...
        'ertugliflozin','steglatro','pramlintide','symlin'};
    b = false;
    lt = lower(strtrim(char(txt)));
    for i = 1:numel(patterns)
        if ~isempty(strfind(lt, patterns{i})), b = true; return; end %#ok<STREMP>
    end
end

% ============================================================ %
function mod = buildNumericModality(src, name, topK, minCoverage)
    % Handles both wide (synthetic: src.X columns already = features) and
    % long (real: src.subject_id/itemid/valuenum) formats.
    if isfield(src, 'itemid')
        mod = pivotLongFormat(src, name, topK, minCoverage);
    else
        % wide: src.subject_id, src.X (n-by-p), src.names
        mod = struct('subject_id', src.subject_id, 'X', src.X, ...
                     'names', {strcat(name, '_', src.names)});
        fprintf("Built '%s': %d samples x %d features (wide)\n", name, ...
                numel(mod.subject_id), numel(mod.names));
    end
end

function mod = pivotLongFormat(src, name, topK, minCoverage)
    % Pivot long-format itemid/valuenum into a sparse patient-by-item matrix,
    % keep items with enough coverage, then mean-impute the remainder. Mirrors
    % _finalize() in modality_builder.py; a hard dropna() here would drop most
    % patients on real sparse EHR panels.
    items = unique(src.itemid);
    counts = zeros(size(items));
    for i = 1:numel(items)
        counts(i) = sum(src.itemid == items(i));
    end
    [~, ord] = sort(counts, 'descend');
    keepItems = items(ord(1:min(topK, numel(items))));

    subs = unique(src.subject_id);
    nSub = numel(subs); nItem = numel(keepItems);
    subjIdx = containers.Map(num2cell(subs), num2cell(1:nSub));
    itemIdx = containers.Map(num2cell(keepItems), num2cell(1:nItem));

    sumMat = zeros(nSub, nItem); cntMat = zeros(nSub, nItem);
    for i = 1:numel(src.subject_id)
        if ~isKey(itemIdx, src.itemid(i)) || isnan(src.valuenum(i)), continue; end
        r = subjIdx(src.subject_id(i)); c = itemIdx(src.itemid(i));
        sumMat(r, c) = sumMat(r, c) + src.valuenum(i);
        cntMat(r, c) = cntMat(r, c) + 1;
    end
    Xpiv = sumMat ./ max(cntMat, 1);
    Xpiv(cntMat == 0) = NaN;

    coverage = mean(cntMat > 0, 1);
    keepCols = coverage >= minCoverage;
    nDropped = sum(~keepCols);
    if nDropped > 0
        fprintf("Modality '%s': dropped %d item column(s) with <%.0f%% coverage\n", ...
                name, nDropped, 100*minCoverage);
    end
    Xpiv = Xpiv(:, keepCols);
    keptItems = keepItems(keepCols);
    if isempty(keptItems)
        mod = struct('subject_id', [], 'X', [], 'names', {{}});
        fprintf("Modality '%s': no columns met coverage threshold - skipped.\n", name);
        return;
    end

    rowHasAny = any(~isnan(Xpiv), 2);
    subs = subs(rowHasAny); Xpiv = Xpiv(rowHasAny, :);

    nImputed = sum(isnan(Xpiv(:)));
    if nImputed > 0
        colMeans = nanmean_local(Xpiv);
        for c = 1:size(Xpiv, 2)
            bad = isnan(Xpiv(:, c));
            Xpiv(bad, c) = colMeans(c);
        end
        fprintf("Modality '%s': mean-imputed %d sparse cell(s)\n", name, nImputed);
    end

    names = arrayfun(@(id) sprintf('%s_item_%d', name, id), keptItems, 'UniformOutput', false);
    mod = struct('subject_id', subs, 'X', Xpiv, 'names', {names});
    fprintf("Built '%s': %d samples x %d features (real, pivoted)\n", name, ...
            numel(subs), numel(names));
end

function m = nanmean_local(X)
    m = zeros(1, size(X,2));
    for c = 1:size(X,2)
        col = X(:,c); col = col(~isnan(col));
        if isempty(col), m(c) = 0; else, m(c) = mean(col); end
    end
end

% ============================================================ %
function mod = buildMedications(src, topK)
    if isfield(src, 'medication')
        % long / real: filter diabetes-drug ROWS before top-k selection,
        % using the robust substring matcher on the RAW free-text name.
        isLeak = false(numel(src.medication), 1);
        for i = 1:numel(src.medication)
            isLeak(i) = isDiabetesDrugName(src.medication{i});
        end
        nLeak = sum(isLeak);
        if nLeak > 0
            fprintf('meds: dropped %d rows naming a diabetes drug before top-k selection\n', nLeak);
        end
        subj = src.subject_id(~isLeak); drug = src.medication(~isLeak);
        uDrugs = unique(drug);
        counts = zeros(size(uDrugs));
        for i = 1:numel(uDrugs), counts(i) = sum(strcmp(drug, uDrugs{i})); end
        [~, ord] = sort(counts, 'descend');
        keepDrugs = uDrugs(ord(1:min(topK, numel(uDrugs))));

        subs = unique(subj); nSub = numel(subs); nDrug = numel(keepDrugs);
        subjIdx = containers.Map(num2cell(subs), num2cell(1:nSub));
        X = zeros(nSub, nDrug);
        for i = 1:numel(subj)
            di = find(strcmp(keepDrugs, drug{i}), 1);
            if isempty(di), continue; end
            X(subjIdx(subj(i)), di) = 1;
        end
        names = cellfun(@(s) ['meds_' strrep(s, ' ', '_')], keepDrugs, 'UniformOutput', false);
        mod = struct('subject_id', subs, 'X', X, 'names', {names});
        fprintf("Built 'meds': %d samples x %d features\n", numel(subs), numel(names));
    else
        % wide / synthetic binary columns
        mod = struct('subject_id', src.subject_id, 'X', src.X, ...
                     'names', {strcat('meds_', src.names)});
        keep = true(1, numel(mod.names));
        for i = 1:numel(mod.names)
            if isDiabetesDrugName(strrep(mod.names{i}, 'meds_', ''))
                keep(i) = false;
            end
        end
        nRemoved = sum(~keep);
        mod.X = mod.X(:, keep); mod.names = mod.names(keep);
        if nRemoved > 0
            fprintf("Modality 'meds': removed %d diabetes-drug leakage cols\n", nRemoved);
        end
        fprintf("Built 'meds': %d samples x %d features (wide)\n", numel(mod.subject_id), numel(mod.names));
    end
end

% ============================================================ %
function mod = buildDiagnoses(src, topK)
    uCodes = unique(src.icd_code);
    counts = zeros(size(uCodes));
    for i = 1:numel(uCodes), counts(i) = sum(strcmp(src.icd_code, uCodes{i})); end
    [~, ord] = sort(counts, 'descend');
    keepCodes = uCodes(ord(1:min(topK, numel(uCodes))));

    subs = unique(src.subject_id); nSub = numel(subs); nCode = numel(keepCodes);
    subjIdx = containers.Map(num2cell(subs), num2cell(1:nSub));
    X = zeros(nSub, nCode);
    for i = 1:numel(src.subject_id)
        ci = find(strcmp(keepCodes, src.icd_code{i}), 1);
        if isempty(ci), continue; end
        X(subjIdx(src.subject_id(i)), ci) = 1;
    end
    % Strip diabetes-defining codes (label leakage)
    keep = true(1, nCode);
    for i = 1:nCode
        if isDiabetesCode(keepCodes{i}), keep(i) = false; end
    end
    nRemoved = sum(~keep);
    X = X(:, keep); keptCodes = keepCodes(keep);
    names = cellfun(@(c) ['dx_' c], keptCodes, 'UniformOutput', false);
    mod = struct('subject_id', subs, 'X', X, 'names', {names});
    fprintf("Built 'dx': %d samples x %d features (removed %d diabetes-code leakage cols)\n", ...
            numel(subs), numel(names), nRemoved);
end

% ============================================================ %
function [X, y, featureNames, modalityMap, subjectIds] = combineModalities(mods, label, how)
    names = fieldnames(mods);
    allSubs = label.subject_id;
    for i = 1:numel(names)
        m = mods.(names{i});
        if strcmpi(how, 'inner')
            allSubs = intersect(allSubs, m.subject_id);
        end
    end
    if strcmpi(how, 'outer')
        allSubs = label.subject_id;  % keep every labelled patient
    end
    allSubs = sort(allSubs);
    n = numel(allSubs);

    X = zeros(n, 0); featureNames = {}; modalityMap = struct();
    col = 0;
    for i = 1:numel(names)
        m = mods.(names{i});
        if isempty(m.names), continue; end
        [~, locB] = ismember(allSubs, m.subject_id);
        block = nan(n, size(m.X, 2));
        found = locB > 0;
        block(found, :) = m.X(locB(found), :);
        if strcmpi(how, 'outer')
            colMeans = nanmean_local(block);
            for c = 1:size(block, 2)
                bad = isnan(block(:, c));
                block(bad, c) = colMeans(c);
            end
        end
        X = [X, block]; %#ok<AGROW>
        featureNames = [featureNames, reshape(m.names, 1, [])]; %#ok<AGROW>
        modalityMap.(names{i}) = (col+1):(col+size(m.X,2));
        col = col + size(m.X, 2);
    end

    [~, locY] = ismember(allSubs, label.subject_id);
    y = label.y(locY);

    if strcmpi(how, 'inner')
        good = all(~isnan(X), 2);
        X = X(good, :); y = y(good); allSubs = allSubs(good);
    end
    subjectIds = allSubs;
    fprintf('Combined (%s): %d samples x %d features, %.1f%% positive\n', ...
            how, numel(y), numel(featureNames), 100*mean(y));
end
