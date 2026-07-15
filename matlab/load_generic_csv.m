function [X, y, names, mmap] = load_generic_csv(csvPath, labelCol, idCol)
% LOAD_GENERIC_CSV  Load an arbitrary tidy CSV (one row per patient) into a
% feature matrix, mirroring the Streamlit app's "Upload my own CSV" path so the
% MATLAB engine can run on datasets that are NOT MIMIC-IV.
%
%   [X, y, names, mmap] = load_generic_csv(csvPath)
%   [X, y, names, mmap] = load_generic_csv(csvPath, labelCol)
%   [X, y, names, mmap] = load_generic_csv(csvPath, labelCol, idCol)
%
% INPUTS
%   csvPath  : path to a .csv with a header row, one row per patient.
%   labelCol : name of the column to predict. Default: 'label'. Must have
%              exactly two distinct values (binary classification).
%   idCol    : (optional) name of a patient-id column to ignore as a feature.
%
% BEHAVIOUR (kept close to src/schema.py:build_matrix_from_mapping)
%   * numeric columns are used as-is and mean-imputed for missing values;
%   * non-numeric (categorical) columns are one-hot encoded;
%   * the label is mapped to 0/1 (second sorted level -> 1).
%
% Portable: uses a manual textscan parse into a struct-of-columns, so it works
% in base MATLAB AND GNU Octave (which has no `table` type).

    if nargin < 2 || isempty(labelCol), labelCol = 'label'; end
    if nargin < 3, idCol = ''; end
    if exist(csvPath, 'file') ~= 2
        error('load_generic_csv:noFile', 'CSV not found: "%s".', csvPath);
    end

    [header, cols] = readCsvColumns(csvPath);   % header: cellstr, cols: cell of column data
    li = find(strcmp(header, labelCol), 1);
    if isempty(li)
        error('load_generic_csv:noLabel', ...
              ['Label column "%s" not found. Columns are: %s. Pass the correct ' ...
               'name as the 2nd argument, e.g. load_generic_csv(path, ''outcome'').'], ...
              labelCol, strjoin(header, ', '));
    end

    % ---- label -> 0/1 ----
    [y, nClasses] = toBinary(cols{li});
    if nClasses ~= 2
        error('load_generic_csv:notBinary', ...
              ['Label column "%s" has %d distinct values; binary classification ' ...
               'needs exactly 2 (e.g. 0/1 or yes/no).'], labelCol, nClasses);
    end

    % ---- features ----
    X = []; names = {};
    for i = 1:numel(header)
        if i == li || strcmp(header{i}, idCol), continue; end
        col = cols{i};
        if isnumeric(col)
            v = col(:);
            mu = mean(v(~isnan(v)));
            if isnan(mu), mu = 0; end
            v(isnan(v)) = mu;
            X = [X, v]; %#ok<AGROW>
            names{end+1} = header{i}; %#ok<AGROW>
        else
            s = cellstr(col);
            levels = unique(s, 'stable');
            for j = 1:numel(levels)
                X = [X, double(strcmp(s, levels{j}))]; %#ok<AGROW>
                names{end+1} = sprintf('%s_%s', header{i}, levels{j}); %#ok<AGROW>
            end
        end
    end

    if isempty(X)
        error('load_generic_csv:noFeatures', ...
              'No usable feature columns found besides the label/id.');
    end
    mmap = struct('uploaded', 1:size(X,2));
    fprintf('load_generic_csv: %d patients x %d features (%d classes) from %s\n', ...
            size(X,1), size(X,2), nClasses, csvPath);
end

% ----------------------------------------------------------------------- %
function [header, cols] = readCsvColumns(path)
% Parse a CSV into a header cellstr and a cell array of columns. Numeric-looking
% columns become double vectors (with NaN for blanks/non-numeric); the rest stay
% as cellstr. Quote-aware (%q) so embedded commas don't break parsing.
    fid = fopen(path, 'r');
    header = strsplit(strtrim(fgetl(fid)), ',');
    nCols = numel(header);
    fmt = repmat('%q', 1, nCols);
    C = textscan(fid, fmt, 'Delimiter', ',', 'CollectOutput', false);
    fclose(fid);
    cols = cell(1, nCols);
    for i = 1:nCols
        raw = C{i};
        if ~iscell(raw), raw = cellstr(string(raw)); end
        num = str2double(raw);
        blanks = strcmp(strtrim(raw), '');
        % treat as numeric if every non-blank entry parses as a number
        if all(~isnan(num) | blanks)
            cols{i} = num;          % blanks -> NaN (mean-imputed later)
        else
            cols{i} = raw;
        end
    end
    header = strtrim(header);
end

% ----------------------------------------------------------------------- %
function [y, nClasses] = toBinary(col)
    if isnumeric(col)
        vals = col(:);
        u = unique(vals(~isnan(vals)));
        nClasses = numel(u);
        if nClasses == 2, y = double(vals == max(u)); else, y = zeros(numel(vals),1); end
    else
        s = cellstr(col);
        u = unique(s, 'stable');
        nClasses = numel(u);
        us = sort(u);
        if nClasses == 2, y = double(strcmp(s, us{2})); else, y = zeros(numel(s),1); end
    end
end
