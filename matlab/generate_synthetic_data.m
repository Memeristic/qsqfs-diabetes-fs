function data = generate_synthetic_data(n, seed)
% GENERATE_SYNTHETIC_DATA  Synthetic MIMIC-IV-shaped demo data.
%
%   data = generate_synthetic_data(n, seed)
%
%   MATLAB port of src/data_loader.py's _generate_synthetic_data. Classes are
%   heavily OVERLAPPING (no single feature separates them), and diabetes drugs
%   and diabetes-defining ICD codes are present in the raw tables but stripped
%   downstream by build_modalities.m, so the label is not baked directly into
%   any single feature.
%
%   OUTPUT  data (struct):
%     .subject_id, .flag (ground-truth, NOT the label used by the pipeline --
%      the label is re-derived from ICD codes exactly as in the real path)
%     .labs        (table-like struct: subject_id + 20 lab columns)
%     .vitals      (subject_id + 5 vitals)
%     .meds        (subject_id + 10 binary medication columns, names in .medNames)
%     .diagnoses   (long format: subject_id, icd_code cell array)
%     .d_icd       (icd_code -> long_title reference)

    if nargin < 1, n = 1000; end
    if nargin < 2, seed = 42; end
    rng(seed, 'twister');

    subject_id = (1:n)';
    anchor_age = randi([20, 89], n, 1);

    risk = 0.018 * (anchor_age - 55) + randn(n, 1);
    prob = 1 ./ (1 + exp(-risk));
    prob = min(max(prob, 0.05), 0.60);
    flag = double(rand(n, 1) < prob);

    % ---- Labs (heavily overlapping distributions) ----
    labNames = {'glucose','hbA1c','creatinine','sodium','potassium','chloride', ...
                'bun','wbc','hemoglobin','platelets','alt','ast','alk_phos', ...
                'bilirubin','albumin','triglycerides','chol_total','hdl','ldl','crp'};
    labs = zeros(n, numel(labNames));
    labs(:,1)  = (128*flag + 104*(1-flag)) + 28*randn(n,1);      % glucose
    labs(:,2)  = (6.6*flag + 5.6*(1-flag))  + 0.9*randn(n,1);    % hbA1c
    labs(:,3)  = 1.0 + 0.3*randn(n,1);                            % creatinine
    labs(:,4)  = 140 + 3*randn(n,1);                              % sodium
    labs(:,5)  = 4.0 + 0.5*randn(n,1);                            % potassium
    labs(:,6)  = 102 + 3*randn(n,1);                              % chloride
    labs(:,7)  = 15 + 5*randn(n,1);                               % bun
    labs(:,8)  = 7.5 + 2*randn(n,1);                              % wbc
    labs(:,9)  = 13.5 + 1.5*randn(n,1);                           % hemoglobin
    labs(:,10) = 250 + 50*randn(n,1);                             % platelets
    labs(:,11) = 35 + 10*randn(n,1);                              % alt
    labs(:,12) = 30 + 8*randn(n,1);                               % ast
    labs(:,13) = 80 + 20*randn(n,1);                              % alk_phos
    labs(:,14) = 0.8 + 0.2*randn(n,1);                            % bilirubin
    labs(:,15) = 4.0 + 0.4*randn(n,1);                            % albumin
    labs(:,16) = (168*flag + 142*(1-flag)) + 45*randn(n,1);      % triglycerides
    labs(:,17) = 200 + 35*randn(n,1);                             % chol_total
    labs(:,18) = 55 + 12*randn(n,1);                              % hdl
    labs(:,19) = 130 + 30*randn(n,1);                             % ldl
    labs(:,20) = (2.6*flag + 2.0*(1-flag)) + 1.4*randn(n,1);     % crp

    % ---- Vitals ----
    vitNames = {'heart_rate','sbp','dbp','resp_rate','temperature'};
    vitals = zeros(n, numel(vitNames));
    vitals(:,1) = 75 + 12*randn(n,1);
    vitals(:,2) = (128*flag + 121*(1-flag)) + 16*randn(n,1);
    vitals(:,3) = 80 + 10*randn(n,1);
    vitals(:,4) = 16 + 3*randn(n,1);
    vitals(:,5) = 36.8 + 0.5*randn(n,1);

    % ---- Medications (diabetes drugs correlate with flag, as in reality;
    %      stripped downstream by build_modalities.m, not here) ----
    medNames = {'metformin','insulin','glipizide','lisinopril','atorvastatin', ...
                'aspirin','amlodipine','metoprolol','omeprazole','furosemide'};
    meds = zeros(n, numel(medNames));
    meds(:,1) = double(rand(n,1) < (0.40*flag + 0.03));
    meds(:,2) = double(rand(n,1) < (0.25*flag + 0.02));
    meds(:,3) = double(rand(n,1) < (0.15*flag + 0.02));
    meds(:,4) = double(rand(n,1) < 0.22);
    meds(:,5) = double(rand(n,1) < 0.20);
    meds(:,6) = double(rand(n,1) < 0.25);
    meds(:,7) = double(rand(n,1) < 0.14);
    meds(:,8) = double(rand(n,1) < 0.16);
    meds(:,9) = double(rand(n,1) < 0.12);
    meds(:,10)= double(rand(n,1) < 0.09);

    % ---- Diagnoses (long format): diabetics -> E11.9, others -> I10;
    %      plus legitimate, non-leaky comorbidity codes. ----
    diag_subject = []; diag_code = {};
    for i = 1:n
        f = flag(i);
        if f, diag_code{end+1} = 'E11.9'; else, diag_code{end+1} = 'I10'; end %#ok<AGROW>
        diag_subject(end+1) = subject_id(i); %#ok<AGROW>
        if rand() < (0.35*f + 0.18)
            diag_subject(end+1) = subject_id(i); diag_code{end+1} = 'I10'; %#ok<AGROW>
        end
        if rand() < (0.30*f + 0.15)
            diag_subject(end+1) = subject_id(i); diag_code{end+1} = 'E78.5'; %#ok<AGROW>
        end
        if rand() < 0.20
            diag_subject(end+1) = subject_id(i); diag_code{end+1} = 'E66.9'; %#ok<AGROW>
        end
    end

    data = struct();
    data.subject_id = subject_id;
    data.flag = flag;
    data.labs = struct('subject_id', subject_id, 'X', labs, 'names', {labNames});
    data.vitals = struct('subject_id', subject_id, 'X', vitals, 'names', {vitNames});
    data.meds = struct('subject_id', subject_id, 'X', meds, 'names', {medNames});
    data.diagnoses = struct('subject_id', diag_subject(:), 'icd_code', {diag_code(:)});
    data.d_icd = struct('icd_code', {{'E11.9','I10','E78.5','E66.9','250.00'}}, ...
                        'long_title', {{'Type 2 diabetes mellitus without complications', ...
                                        'Essential (primary) hypertension', ...
                                        'Hyperlipidemia, unspecified', ...
                                        'Obesity, unspecified', ...
                                        'Diabetes mellitus without complication (ICD-9)'}});
    fprintf('Synthetic data: n=%d, prevalence=%.1f%% (overlapping signal)\n', n, 100*mean(flag));
end
