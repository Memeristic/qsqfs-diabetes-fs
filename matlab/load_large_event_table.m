function T = load_large_event_table(path, top_k, chunk_rows)
% LOAD_LARGE_EVENT_TABLE  Scalable, chunked read of a long-format MIMIC-IV
% event table (labevents.csv or chartevents.csv: subject_id, itemid, valuenum
% [+ other columns, ignored]).
%
%   T = load_large_event_table(path, top_k, chunk_rows)
%
% MATLAB/Octave port of the same fix applied in src/data_loader.py
% (`_load_large_event_table`): real MIMIC-IV labevents (~120M rows) and
% chartevents (~330M rows) cannot be read whole into memory. This streams the
% file in fixed-size row chunks via fopen/textscan (portable to both MATLAB
% and GNU Octave -- no toolbox required):
%   Pass 1: stream only the itemid column to get true frequency counts.
%   Pass 2: stream subject_id/itemid/valuenum and keep only rows whose itemid
%           is among the most frequent `top_k` (buffered to 3x for headroom).
%
% OUTPUT: T.subject_id, T.itemid, T.valuenum (numeric column vectors).

    if nargin < 1 || isempty(path) || ~ischar(path)
        error('load_large_event_table:badPath', 'A CSV file path is required.');
    end
    if nargin < 2 || isempty(top_k), top_k = 50; end
    if nargin < 3 || isempty(chunk_rows), chunk_rows = 2000000; end
    if ~(isnumeric(top_k) && isscalar(top_k) && top_k >= 1)
        error('load_large_event_table:badTopK', 'top_k must be a positive integer (got %s).', mat2str(top_k));
    end
    if exist(path, 'file') ~= 2
        error('load_large_event_table:noFile', 'Event table not found: "%s".', path);
    end
    if numel(path) >= 3 && strcmpi(path(end-2:end), '.gz')
        error('load_large_event_table:gzip', ...
              ['"%s" is gzip-compressed. Decompress it first (gunzip "%s") -- the portable, ' ...
               'toolbox-free reader uses fopen/textscan and cannot stream .gz directly.'], path, path);
    end
    firstLine = getFirstLine(path);
    if ~ischar(firstLine) || isempty(strtrim(firstLine))
        error('load_large_event_table:empty', 'Event table "%s" is empty or has no header row.', path);
    end
    header = strsplit(strtrim(firstLine), ',');
    idxSubj = findCol(header, 'subject_id');
    idxItem = findCol(header, 'itemid');
    idxVal  = findCol(header, 'valuenum');
    if any([idxSubj, idxItem, idxVal] == 0)
        error('load_large_event_table:missingCols', ...
              'Required columns subject_id/itemid/valuenum not found in %s', path);
    end
    nCols = numel(header);

    % ---- Pass 1: itemid frequency ----
    fprintf('%s: pass 1/2 - streaming itemid frequency...\n', path);
    counts = containers.Map('KeyType', 'double', 'ValueType', 'double');
    fid = fopen(path, 'r');
    fgetl(fid);  % header
    % Use %q (quote-aware), not %s: real MIMIC-IV event tables have
    % free-text columns (e.g. labevents.comments) containing embedded commas and
    % doubled quotes. Plain %s + ',' delimiter mis-splits those rows and desyncs
    % the whole file; %q respects RFC-4180 quoting. (%q works in both MATLAB and
    % Octave.) Verified against the real MIMIC-IV demo (labevents.comments has
    % thousands of comma-bearing quoted cells).
    fmt = repmat('%q', 1, nCols);
    while ~feof(fid)
        C = textscan(fid, fmt, chunk_rows, 'Delimiter', ',', 'HeaderLines', 0);
        if isempty(C{1}), break; end
        items = str2double(C{idxItem});
        items = items(~isnan(items));
        u = unique(items);
        cnts = histc(items, u); %#ok<HISTC>
        for k = 1:numel(u)
            key = u(k);
            if isKey(counts, key), counts(key) = counts(key) + cnts(k);
            else, counts(key) = cnts(k); end
        end
    end
    fclose(fid);

    allItems = cell2mat(keys(counts));
    allCounts = cell2mat(values(counts));
    [~, ord] = sort(allCounts, 'descend');
    keepN = min(numel(allItems), max(top_k * 3, top_k + 10));
    topItems = allItems(ord(1:keepN));
    topSet = containers.Map(num2cell(topItems), num2cell(true(size(topItems))));
    fprintf('%s: keeping top %d itemids by frequency (of %d distinct)\n', ...
            path, numel(topItems), numel(allItems));

    % ---- Pass 2: filtered read ----
    fprintf('%s: pass 2/2 - streaming filtered rows...\n', path);
    subj_parts = {}; item_parts = {}; val_parts = {};
    fid = fopen(path, 'r');
    fgetl(fid);
    fmt = repmat('%q', 1, nCols);   % quote-aware (see pass 1 note)
    while ~feof(fid)
        C = textscan(fid, fmt, chunk_rows, 'Delimiter', ',', 'HeaderLines', 0);
        if isempty(C{1}), break; end
        subj = str2double(C{idxSubj});
        item = str2double(C{idxItem});
        val  = str2double(C{idxVal});
        keep = false(size(item));
        for i = 1:numel(item)
            keep(i) = ~isnan(item(i)) && isKey(topSet, item(i));
        end
        subj_parts{end+1} = subj(keep); %#ok<AGROW>
        item_parts{end+1} = item(keep); %#ok<AGROW>
        val_parts{end+1}  = val(keep);  %#ok<AGROW>
    end
    fclose(fid);

    T = struct();
    if isempty(subj_parts)
        T.subject_id = zeros(0,1); T.itemid = zeros(0,1); T.valuenum = zeros(0,1);
        warning('load_large_event_table:noRows', ...
                ['%s: no rows survived itemid filtering, so this modality will be empty. ' ...
                 'Usually this means top_k is far too small, the file covers a different set of ' ...
                 'patients, or the wrong column was used as itemid.'], path);
        return;
    end
    T.subject_id = cat(1, subj_parts{:});
    T.itemid     = cat(1, item_parts{:});
    T.valuenum   = cat(1, val_parts{:});
    fprintf('%s: %d rows retained after itemid filtering (scalable path)\n', ...
            path, numel(T.subject_id));
end

function line = getFirstLine(path)
    fid = fopen(path, 'r');
    line = fgetl(fid);
    fclose(fid);
end

function idx = findCol(header, name)
    idx = find(strcmpi(strtrim(header), name), 1);
    if isempty(idx), idx = 0; end
end
