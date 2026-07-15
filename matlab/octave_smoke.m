function octave_smoke()
% OCTAVE_SMOKE  Fast sanity check for the MATLAB/Octave port, suitable for CI.
% Exercises the portable pieces (no GUI, no heavy pipeline): the generic CSV
% loader and run_pipeline's argument dispatch/guards. Exits non-zero on failure
% so a CI job can gate on it. Run: octave-cli --eval "octave_smoke"
    addpath(fileparts(mfilename('fullpath')));
    failures = 0;

    % --- 1. load_generic_csv on a tiny inline CSV ---
    tmp = tempname(); fid = fopen(tmp, 'w');
    fprintf(fid, 'id,f1,f2,cat,label\n');
    rows = {1,0.5,1.2,'a',1; 2,-0.3,0.8,'b',0; 3,1.1,-0.4,'a',1; ...
            4,0.2,0.1,'b',0; 5,-0.9,1.5,'a',1; 6,0.7,-0.2,'b',0};
    for i = 1:size(rows,1)
        fprintf(fid, '%d,%.2f,%.2f,%s,%d\n', rows{i,1}, rows{i,2}, rows{i,3}, rows{i,4}, rows{i,5});
    end
    fclose(fid);
    try
        [X, y, names, mmap] = load_generic_csv(tmp, 'label', 'id');
        assert(size(X,1) == 6, 'row count');
        assert(numel(unique(y)) == 2, 'binary label');
        assert(any(strcmp(names, 'cat_a')) && any(strcmp(names, 'cat_b')), 'one-hot');
        assert(numel(fieldnames(mmap)) == 1, 'single modality');
        fprintf('PASS  load_generic_csv (%dx%d, %d classes)\n', size(X,1), size(X,2), numel(unique(y)));
    catch e
        fprintf('FAIL  load_generic_csv: %s\n', e.message); failures = failures + 1;
    end
    delete(tmp);

    % --- 2. non-binary label must be rejected ---
    tmp2 = tempname(); fid = fopen(tmp2, 'w');
    fprintf(fid, 'f1,label\n1,0\n2,1\n3,2\n'); fclose(fid);
    try
        load_generic_csv(tmp2, 'label');
        fprintf('FAIL  non-binary label was not rejected\n'); failures = failures + 1;
    catch
        fprintf('PASS  non-binary label rejected\n');
    end
    delete(tmp2);

    % --- 3. run_pipeline argument guards ---
    try
        run_pipeline('csv');   % missing path -> must error
        fprintf('FAIL  run_pipeline(''csv'') did not guard missing path\n'); failures = failures + 1;
    catch
        fprintf('PASS  run_pipeline argument guard\n');
    end

    if failures > 0
        error('octave_smoke:failed', '%d smoke check(s) failed.', failures);
    end
    fprintf('ALL OCTAVE SMOKE CHECKS PASSED\n');
end
