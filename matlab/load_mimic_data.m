function data = load_mimic_data(dataRoot, cfg)
% LOAD_MIMIC_DATA  Load a real MIMIC-IV directory into the struct shape
% build_modalities.m expects.
%
%   data = load_mimic_data(dataRoot, cfg)
%
% dataRoot must contain:
%   hosp/{patients,diagnoses_icd,d_icd_diagnoses,labevents,d_labitems,
%         pharmacy}.csv[.gz]
%   icu/{chartevents,d_items}.csv[.gz]
% (matches the official MIMIC-IV download layout exactly)
%
% Small reference tables use plain fopen/textscan (portable, no toolbox).
% labevents/chartevents use load_large_event_table.m's chunked, frequency-
% filtered reader (see that file for why -- real MIMIC-IV is 100-330M+ rows
% per table and cannot be read whole into memory).
%
% cfg (optional): .top_labs (50), .top_vitals (30) -- passed through to the
% large-table reader as the itemid-frequency keep budget.

    if nargin < 1 || isempty(dataRoot) || ~ischar(dataRoot)
        error('load_mimic_data:badPath', ...
              ['dataRoot must be a folder path, e.g. ' ...
               'load_mimic_data(''/path/to/mimic-iv''). See matlab/README.md.']);
    end
    if nargin < 2, cfg = struct(); end
    topLabs = getf(cfg, 'top_labs', 50);
    topVitals = getf(cfg, 'top_vitals', 30);

    if exist(dataRoot, 'dir') ~= 7
        error('load_mimic_data:noRoot', ...
              'MIMIC-IV folder not found: "%s". Point run_pipeline at the folder that contains hosp/ and icu/.', dataRoot);
    end
    hospDir = fullfile(dataRoot, 'hosp');
    icuDir = fullfile(dataRoot, 'icu');
    if exist(hospDir, 'dir') ~= 7
        error('load_mimic_data:noHosp', ...
              ['Expected a "hosp" subfolder inside "%s" but none was found. Lay the data out exactly ' ...
               'as it ships from PhysioNet:\n  %s/hosp/{patients,diagnoses_icd,d_icd_diagnoses,labevents,d_labitems,pharmacy}.csv\n' ...
               '  %s/icu/{chartevents,d_items}.csv\nand decompress any .csv.gz first (gunzip *.csv.gz).'], ...
              dataRoot, dataRoot, dataRoot);
    end
    if exist(icuDir, 'dir') ~= 7
        fprintf(['Note: no "icu" subfolder under %s -- the vitals (chartevents) modality ' ...
                 'will be skipped. This is fine; labs/meds/dx can still run.\n'], dataRoot);
    end

    data = struct();

    % ---- diagnoses_icd (long: subject_id, icd_code) ----
    p = resolvePath(hospDir, 'diagnoses_icd');
    if ~isempty(p)
        [subj, code] = readTwoColCsv(p, 'subject_id', 'icd_code');
        data.diagnoses = struct('subject_id', subj, 'icd_code', {code});
        fprintf('Loaded diagnoses_icd: %d rows\n', numel(subj));
    else
        error('load_mimic_data:missing', 'diagnoses_icd.csv[.gz] not found under %s', hospDir);
    end

    % ---- labevents (large, long: subject_id, itemid, valuenum) ----
    p = resolvePath(hospDir, 'labevents');
    if ~isempty(p)
        data.labs = load_large_event_table(p, topLabs);
    else
        fprintf('labevents.csv not found - skipping labs modality.\n');
    end

    % ---- chartevents (large, long: subject_id, itemid, valuenum) ----
    p = resolvePath(icuDir, 'chartevents');
    if ~isempty(p)
        data.vitals = load_large_event_table(p, topVitals);
    else
        fprintf('chartevents.csv not found - skipping vitals modality.\n');
    end

    % ---- pharmacy (long: subject_id, medication) ----
    p = resolvePath(hospDir, 'pharmacy');
    if isempty(p), p = resolvePath(hospDir, 'prescriptions'); end
    if ~isempty(p)
        [subj, med] = readTwoColCsv(p, 'subject_id', {'medication', 'drug'});
        data.meds = struct('subject_id', subj, 'medication', {med});
        fprintf('Loaded pharmacy/prescriptions: %d rows\n', numel(subj));
    else
        fprintf('pharmacy/prescriptions.csv not found - skipping meds modality.\n');
    end

    % ---- reference tables for clinical-alignment mapping (optional) ----
    data.d_labitems = readItemLabels(resolvePath(hospDir, 'd_labitems'));
    data.d_items    = readItemLabels(resolvePath(icuDir, 'd_items'));
end

% ============================================================ %
function v = getf(s, name, default)
    if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end

function p = resolvePath(dirPath, stem)
    cands = {fullfile(dirPath, [stem '.csv']), fullfile(dirPath, [stem '.csv.gz'])};
    p = '';
    for i = 1:numel(cands)
        if exist(cands{i}, 'file'), p = cands{i}; return; end
    end
end

function header = readHeader(path)
    if endsWithStr(path, '.gz')
        error('load_mimic_data:gzip', ...
              'Gzip-compressed CSVs must be decompressed first (gunzip %s) -- Octave/base MATLAB textscan cannot stream .gz directly.', path);
    end
    fid = fopen(path, 'r');
    header = strsplit(strtrim(fgetl(fid)), ',');
    fclose(fid);
end

function b = endsWithStr(s, suf)
    b = numel(s) >= numel(suf) && strcmp(s(end-numel(suf)+1:end), suf);
end

function [subj, textCol] = readTwoColCsv(path, subjName, textNameOrList)
    header = readHeader(path);
    idxSubj = find(strcmpi(strtrim(header), subjName), 1);
    if ischar(textNameOrList), textNameOrList = {textNameOrList}; end
    idxText = 0;
    for i = 1:numel(textNameOrList)
        f = find(strcmpi(strtrim(header), textNameOrList{i}), 1);
        if ~isempty(f), idxText = f; break; end
    end
    if isempty(idxSubj) || idxText == 0
        error('load_mimic_data:missingCols', 'Required columns not found in %s', path);
    end
    fmt = repmat('%q', 1, numel(header));   % quote-aware: real MIMIC free-text has embedded commas/quotes
    fid = fopen(path, 'r'); fgetl(fid);
    C = textscan(fid, fmt, 'Delimiter', ',');
    fclose(fid);
    subj = str2double(C{idxSubj});
    textCol = C{idxText};
end

function tbl = readItemLabels(path)
    tbl = struct('itemid', [], 'label', {{}});
    if isempty(path), return; end
    header = readHeader(path);
    idxId = find(strcmpi(strtrim(header), 'itemid'), 1);
    idxLbl = find(strcmpi(strtrim(header), 'label'), 1);
    if isempty(idxId) || isempty(idxLbl), return; end
    fmt = repmat('%q', 1, numel(header));   % quote-aware: real MIMIC free-text has embedded commas/quotes
    fid = fopen(path, 'r'); fgetl(fid);
    C = textscan(fid, fmt, 'Delimiter', ',');
    fclose(fid);
    tbl.itemid = str2double(C{idxId});
    tbl.label = C{idxLbl};
end
