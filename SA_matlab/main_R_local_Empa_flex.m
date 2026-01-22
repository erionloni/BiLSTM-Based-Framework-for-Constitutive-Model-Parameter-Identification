function main_R_local_Empa_flex
% Flexible relaxation simulations of compressible Rubin-Bodner model.
% Supports single-rate ramps, ramp+hold, multi-step piecewise ramps, and
% multi-rate repetitions (same specimen tested at several constant rates).

clearvars;
clc;
addpath(genpath(pwd));

% Parameter file (can be overridden via environment variable)
param_file = getenv('PARAM_FILE');
if isempty(param_file)
    param_file = 'D2.txt';
end

solver_dl = str2double(getenv('SOLVER_DL'));
if isnan(solver_dl) || solver_dl <= 0
    solver_dl = 0.001; % default solver lambda increment
end

start_idx = str2double(getenv('START_IDX'));
if isnan(start_idx) || start_idx < 1
    start_idx = 1;
    warning('START_IDX not set, defaulting to 1');
end

% Load parameters
parameters = readmatrix(param_file);
if size(parameters, 2) < 12
    error('Parameter file must have at least 12 columns');
end
n_cases = size(parameters, 1);

end_idx = str2double(getenv('END_IDX'));
if isnan(end_idx) || end_idx < start_idx
    end_idx = n_cases;
    warning('END_IDX not set, defaulting to %d', end_idx);
end
end_idx = min(end_idx, n_cases);

fprintf('\n========================================\n');
fprintf('MATLAB Flexible Batch Processing\n');
fprintf('========================================\n');
fprintf('Parameter file: %s\n', param_file);
fprintf('Processing cases: %d to %d (of %d)\n', start_idx, end_idx, n_cases);
fprintf('========================================\n\n');

% Base parameters (columns 1-12)
lambda1_max_all = get_col(parameters, 1, 1)';
theta_all       = get_col(parameters, 2, 0)';
q_all           = get_col(parameters, 3, 0)';
m1_all          = get_col(parameters, 4, 0)';
m2_all          = get_col(parameters, 5, 0)';
m3_all          = get_col(parameters, 6, 0)';
m4_all          = get_col(parameters, 7, 0)';
m5_all          = get_col(parameters, 8, 0)';
kM_all          = get_col(parameters, 9, 0)';
alphaM_all      = get_col(parameters,10, 0)';
kF_all          = get_col(parameters,11, 0)';
lambda1_dot_all = get_col(parameters,12, 0)';

% Flexible loading fields
scenario_type_all = get_col(parameters,13, 0)';  % 0=ramp,1=hold,2=multi-step,3=multi-rate
n_segments_all    = get_col(parameters,14, 0)';
seg1_lambda_all   = get_col(parameters,15, 0)';
seg1_rate_all     = get_col(parameters,16, 0)';
seg2_lambda_all   = get_col(parameters,17, 0)';
seg2_rate_all     = get_col(parameters,18, 0)';
seg3_lambda_all   = get_col(parameters,19, 0)';
seg3_rate_all     = get_col(parameters,20, 0)';
hold_time_all     = get_col(parameters,21, 0)';

% Multi-rate additions (scenario_type == 3)
max_multi_rate_cols = max(0, size(parameters, 2) - 22);
multi_rate_count_all = get_col(parameters,22, 0)';
multi_rates_all = zeros(n_cases, max(1, max_multi_rate_cols));
for idx = 1:max_multi_rate_cols
    multi_rates_all(:, idx) = get_col(parameters, 22 + idx, 0);
end

fprintf('Loaded %d parameter sets (%d columns)\n\n', size(parameters, 1), size(parameters, 2));

% Output directory (can be overridden via environment variable)
output_subdir = getenv('OUTPUT_SUBDIR');
if isempty(output_subdir)
    output_subdir = 'matlab_data';  % default
end
if ~exist(output_subdir, 'dir')
    mkdir(output_subdir);
end
fprintf('Output directory: %s\n', output_subdir);

% Constants
parE.NF = 16; % number of fiber directions

% =========================================================================
% PARALLEL PROCESSING SETUP
% =========================================================================
% Configure pool via environment variables (useful on Slurm):
%   PARPOOL_MODE     = threads | local | none   (default: threads on Slurm, local otherwise)
%   PARPOOL_WORKERS  = integer > 0             (default: SLURM_CPUS_PER_TASK, else MATLAB default)
parpool_mode = getenv('PARPOOL_MODE');
if isempty(parpool_mode)
    if ~isempty(getenv('SLURM_JOB_ID'))
        parpool_mode = 'threads';
    else
        parpool_mode = 'local';
    end
end

parpool_workers = str2double(getenv('PARPOOL_WORKERS'));
if isnan(parpool_workers) || parpool_workers <= 0
    parpool_workers = str2double(getenv('SLURM_CPUS_PER_TASK'));
end
if isnan(parpool_workers) || parpool_workers <= 0
    parpool_workers = 0; % MATLAB default
end

% Prepare indices for processing (used to decide if parallelism is worth it)
case_indices = start_idx:end_idx;
n_to_process = numel(case_indices);

use_parallel = n_to_process > 1 && ~strcmpi(parpool_mode, 'none');

% Start parallel pool if requested and not already running
if use_parallel && isempty(gcp('nocreate'))
    fprintf('Starting parallel pool (%s)...\n', parpool_mode);
    try
        if strcmpi(parpool_mode, 'threads')
            if parpool_workers > 0
                parpool('threads', parpool_workers);
            else
                parpool('threads');
            end
        else
            if parpool_workers > 0
                parpool('local', parpool_workers);  % process-based workers
            else
                parpool('local');
            end
        end
    catch poolErr
        warning('Failed to start parpool (%s). Falling back to sequential execution: %s', parpool_mode, poolErr.message);
        use_parallel = false;
    end
end

if use_parallel
    fprintf('Starting simulations with parallel processing...\n\n');
else
    fprintf('Starting simulations (sequential)...\n\n');
end
batch_start_time = tic;

% Preallocate cell arrays for parallel results
results = cell(n_to_process, 1);
success_flags = false(n_to_process, 1);
case_ids = zeros(n_to_process, 1);

% =========================================================================
% MAIN PARALLEL LOOP
% =========================================================================
if use_parallel
    parfor pp = 1:n_to_process
        ll = case_indices(pp);
        case_ids(pp) = ll;
    
        % Extract parameters for this case
        scenario = scenario_type_all(ll);
        current_hold_time = hold_time_all(ll);
        lambda1_max = lambda1_max_all(ll);
        lambda1_dot = lambda1_dot_all(ll);
    
        % Assemble parameter vector (per case)
        par = zeros(13, 1);
        par(1)  = kM_all(ll);
        par(2)  = kF_all(ll);
        par(3)  = 1;          % mu0
        par(4)  = m3_all(ll);
        par(5)  = m5_all(ll);
        par(6)  = q_all(ll);
        par(7)  = m1_all(ll);
        par(8)  = m2_all(ll);
        par(9)  = m4_all(ll);
        par(10) = theta_all(ll);
        par(11) = alphaM_all(ll);
    
    % =====================================================================
    % Parameter validation
    % =====================================================================
    if any(isnan(par)) || any(isinf(par))
        fprintf('Case %d: NaN/Inf parameters detected, skipping\n', ll);
        success_flags(pp) = false;
        results{pp} = struct('error', 'NaN/Inf parameters');
        continue;
    end
    if par(9) <= 1  % m4 must be > 1
        fprintf('Case %d: Warning - m4 = %.4f <= 1, may cause issues\n', ll, par(9));
    end
    if par(1) <= 0 || par(2) <= 0  % kM, kF must be positive
        fprintf('Case %d: Warning - kM or kF <= 0\n', ll);
    end
    
    case_failed = false;
    out_struct = struct();
    
    if scenario == 3
        % Multi-rate repetition: run independent ramps from reference for each rate
        n_rates = max(1, min(max_multi_rate_cols, round(multi_rate_count_all(ll))));
        if max_multi_rate_cols > 0
            rate_list = multi_rates_all(ll, 1:max(1, n_rates));
        else
            rate_list = lambda1_dot;
        end
        if isempty(rate_list) || all(rate_list == 0)
            rate_list = lambda1_dot;
        end
        rate_list(rate_list <= 0) = lambda1_dot;
        
        curve_cells = cell(1, numel(rate_list));
        
        for rIdx = 1:numel(rate_list)
            rate_val = rate_list(rIdx);
            [lambda1, runTime, eff_rate, used_rates] = build_loading_schedule( ...
                lambda1_max, rate_val, 0, 0, [0, 0, 0], [0, 0, 0], current_hold_time, solver_dl);
            
            par(12) = eff_rate;
            par(13) = lambda1(end);
            
            [curve, ok] = simulate_curve(lambda1, runTime, parE, par, 0, current_hold_time);
            if ~ok
                case_failed = true;
                break;
            end
            
            curve.segment_rates = used_rates;
            curve.scenario = scenario;
            curve.rate = rate_val;
            curve.hold_time = current_hold_time;
            curve_cells{rIdx} = curve;
        end
        
        if case_failed
            fprintf('Case %d: Solver failure in multi-rate simulation\n', ll);
            success_flags(pp) = false;
            results{pp} = struct('error', 'Solver failure');
            continue;
        end
        
        out_struct = struct();
        out_struct.multi_rates = rate_list(:);
        out_struct.multi_curves = curve_cells;
        out_struct.par = curve_cells{1}.par;
        out_struct.scenario = scenario;
        out_struct.hold_time = current_hold_time;
    else
        % Extract segment data for this case
        seg_lambdas = [seg1_lambda_all(ll), seg2_lambda_all(ll), seg3_lambda_all(ll)];
        seg_rates = [seg1_rate_all(ll), seg2_rate_all(ll), seg3_rate_all(ll)];
        n_segments = round(n_segments_all(ll));
        
        [lambda1, runTime, eff_rate, used_rates] = build_loading_schedule( ...
            lambda1_max, lambda1_dot, scenario, n_segments, ...
            seg_lambdas, seg_rates, current_hold_time, solver_dl);
        
        par(12) = eff_rate;
        par(13) = lambda1(end);
        
        [curve, ok] = simulate_curve(lambda1, runTime, parE, par, scenario, current_hold_time);
        if ~ok
            fprintf('Case %d: Solver failure\n', ll);
            success_flags(pp) = false;
            results{pp} = struct('error', 'Solver failure');
            continue;
        end
        
        curve.segment_rates = used_rates;
        curve.scenario = scenario;
        curve.hold_time = current_hold_time;
        out_struct = curve;
    end
    
    % Store result
    results{pp} = out_struct;
    success_flags(pp) = true;
    
    % =====================================================================
    % INCREMENTAL SAVE (Inside parfor)
    % =====================================================================
    % Save immediately so files appear during the run
        save_path = sprintf('%s/data_matlab_augmented_case_%d.mat', output_subdir, ll);
        par_save(save_path, out_struct);
    end
else
    for pp = 1:n_to_process
        ll = case_indices(pp);
        case_ids(pp) = ll;
        
        % Extract parameters for this case
        scenario = scenario_type_all(ll);
        current_hold_time = hold_time_all(ll);
        lambda1_max = lambda1_max_all(ll);
        lambda1_dot = lambda1_dot_all(ll);
        
        % Assemble parameter vector (per case)
        par = zeros(13, 1);
        par(1)  = kM_all(ll);
        par(2)  = kF_all(ll);
        par(3)  = 1;          % mu0
        par(4)  = m3_all(ll);
        par(5)  = m5_all(ll);
        par(6)  = q_all(ll);
        par(7)  = m1_all(ll);
        par(8)  = m2_all(ll);
        par(9)  = m4_all(ll);
        par(10) = theta_all(ll);
        par(11) = alphaM_all(ll);
        
        if any(isnan(par)) || any(isinf(par))
            fprintf('Case %d: NaN/Inf parameters detected, skipping\n', ll);
            success_flags(pp) = false;
            results{pp} = struct('error', 'NaN/Inf parameters');
            continue;
        end
        if par(9) <= 1
            fprintf('Case %d: Warning - m4 = %.4f <= 1, may cause issues\n', ll, par(9));
        end
        if par(1) <= 0 || par(2) <= 0
            fprintf('Case %d: Warning - kM or kF <= 0\n', ll);
        end
        
        case_failed = false;
        out_struct = struct();
        
        if scenario == 3
            n_rates = max(1, min(max_multi_rate_cols, round(multi_rate_count_all(ll))));
            if max_multi_rate_cols > 0
                rate_list = multi_rates_all(ll, 1:max(1, n_rates));
            else
                rate_list = lambda1_dot;
            end
            if isempty(rate_list) || all(rate_list == 0)
                rate_list = lambda1_dot;
            end
            rate_list(rate_list <= 0) = lambda1_dot;
            
            curve_cells = cell(1, numel(rate_list));
            
            for rIdx = 1:numel(rate_list)
                rate_val = rate_list(rIdx);
                [lambda1, runTime, eff_rate, used_rates] = build_loading_schedule( ...
                    lambda1_max, rate_val, 0, 0, [0, 0, 0], [0, 0, 0], current_hold_time, solver_dl);
                
                par(12) = eff_rate;
                par(13) = lambda1(end);
                
                [curve, ok] = simulate_curve(lambda1, runTime, parE, par, 0, current_hold_time);
                if ~ok
                    case_failed = true;
                    break;
                end
                
                curve.segment_rates = used_rates;
                curve.scenario = scenario;
                curve.rate = rate_val;
                curve.hold_time = current_hold_time;
                curve_cells{rIdx} = curve;
            end
            
            if case_failed
                fprintf('Case %d: Solver failure in multi-rate simulation\n', ll);
                success_flags(pp) = false;
                results{pp} = struct('error', 'Solver failure');
                continue;
            end
            
            out_struct.multi_rates = rate_list;
            out_struct.multi_curves = curve_cells;
            out_struct.scenario = scenario;
            out_struct.par = par;
            out_struct.hold_time = current_hold_time;
        else
            [lambda1, runTime, eff_rate, used_rates] = build_loading_schedule( ...
                lambda1_max, lambda1_dot, scenario, round(n_segments_all(ll)), ...
                [seg1_lambda_all(ll), seg2_lambda_all(ll), seg3_lambda_all(ll)], ...
                [seg1_rate_all(ll), seg2_rate_all(ll), seg3_rate_all(ll)], ...
                current_hold_time, solver_dl);
            
            par(12) = eff_rate;
            par(13) = lambda1(end);
            
            [curve, ok] = simulate_curve(lambda1, runTime, parE, par, scenario, current_hold_time);
            if ~ok
                fprintf('Case %d: Solver failure\n', ll);
                success_flags(pp) = false;
                results{pp} = struct('error', 'Solver failure');
                continue;
            end
            
            curve.segment_rates = used_rates;
            curve.scenario = scenario;
            curve.hold_time = current_hold_time;
            out_struct = curve;
        end
        
        results{pp} = out_struct;
        success_flags(pp) = true;
        
        save_path = sprintf('%s/data_matlab_augmented_case_%d.mat', output_subdir, ll);
        par_save(save_path, out_struct);
    end
end

% =========================================================================
% SAVE RESULTS (Sequential - after parfor)
% =========================================================================
% (Removed: Saving is now done incrementally inside parfor)
failed_cases = case_ids(~success_flags);

% Summary
total_time = toc(batch_start_time);
num_failed = numel(failed_cases);
num_successful = n_to_process - num_failed;

fprintf('\n========================================\n');
fprintf('Batch completed!\n');
fprintf('Cases processed: %d to %d (%d total)\n', start_idx, end_idx, n_to_process);
fprintf('Successful: %d | Failed: %d (%.1f%% success rate)\n', num_successful, num_failed, ...
    100 * num_successful / n_to_process);
if num_failed > 0
    fprintf('Failed case indices: ');
    if num_failed <= 20
        fprintf('%d ', failed_cases);
    else
        fprintf('%d ', failed_cases(1:10));
        fprintf('... and %d more', num_failed - 10);
    end
    fprintf('\n');
end
fprintf('Total time: %.2f minutes (%.2f hours)\n', total_time / 60, total_time / 3600);
fprintf('Average time per case: %.2f seconds\n', total_time / n_to_process);
fprintf('========================================\n');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELPER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function par_save(fname, data)
    save(fname, '-struct', 'data');
end

function col = get_col(mat, idx, default_val)
% Safe column accessor with default fill.
    if size(mat, 2) >= idx
        col = mat(:, idx);
    else
        col = default_val * ones(size(mat, 1), 1);
    end
end

function [curve, success] = simulate_curve(lambda1, runTime, parE, par, scenario, hold_time)
% Run one loading schedule and return the stress/strain curves.
% Handles hold phases and uses computed F11 consistently.

    success = true;
    curve = struct(); % Initialize to satisfy output requirement
    lmax = numel(lambda1);
    rTmax = numel(runTime);

    if lmax ~= rTmax
        warning('lambda1 and time vectors differ in length; truncating to min length');
        max_len = min(lmax, rTmax);
        lambda1 = lambda1(1:max_len);
        runTime = runTime(1:max_len);
        lmax = max_len;
        rTmax = max_len;
    end

    % =====================================================================
    % Verify time vector is monotonic
    % =====================================================================
    if any(diff(runTime) <= 0)
        warning('Time vector is not strictly monotonic');
        % Fix by adding small epsilon to duplicates
        for ti = 2:numel(runTime)
            if runTime(ti) <= runTime(ti-1)
                runTime(ti) = runTime(ti-1) + 1e-9;
            end
        end
    end

    % Preallocate
    F = cell(1, lmax);
    b = cell(1, lmax);
    J = cell(1, lmax);
    Je = cell(1, lmax);
    be = cell(1, lmax);
    s = cell(1, lmax);
    g_save = cell(1, lmax);
    sM_save = cell(1, lmax);
    sN_save = cell(1, lmax);
    sF_save = cell(1, lmax);
    me = cell(parE.NF, lmax);
    ae = cell(parE.NF, lmax);

    % Initialize reference state
    F{1} = eye(3);
    b{1} = F{1} * F{1}';
    be{1} = b{1};
    J{1} = 1;
    Je{1} = 1;
    s{1} = zeros(3, 3);
    g_save{1} = 0;
    sM_save{1} = zeros(3, 3);
    sN_save{1} = zeros(3, 3);
    sF_save{1} = zeros(3, 3);

    % Fiber directions at reference
    for k = 1:parE.NF
        phi = (pi / parE.NF) * (k - 1.5);
        me{k, 1} = [cos(phi) * sin(0.5 * pi - par(10));
                    sin(phi) * sin(0.5 * pi - par(10));
                    ((-1) ^ k) * cos(0.5 * pi - par(10))];
        ae{k, 1} = norm(me{k, 1});
    end

    % Initialization values for constants
    kappa = 1;
    nu = ones(1, parE.NF);

    % Initialization values for lambda increments
    dl2 = 0;
    dl3 = 0;

    for i = 2:rTmax
        dl1 = lambda1(i) - lambda1(i - 1);
        dt = runTime(i) - runTime(i - 1);
        
        % Safety check for dt
        if dt <= 0
            dt = 1e-9;
        end

        kappa_old = kappa;
        nu_old = nu;

        x0 = [dl2, dl3, kappa, nu];

        % Previous tensors
        F_old = F{i - 1};
        be_old = be{i - 1};
        me_old = cell(parE.NF, 1);
        for k = 1:parE.NF
            me_old{k} = me{k, i - 1};
        end

        options = optimoptions('lsqnonlin', ...
            'FunctionTolerance', 1e-12, ...
            'StepTolerance', 1e-12, ...
            'Display', 'none', ...
            'MaxIterations', 1000, ...
            'MaxFunctionEvaluations', 5000);
        lb = -Inf * ones(1, 3 + parE.NF);
        ub = Inf * ones(1, 3 + parE.NF);
        
        try
            [x, ~, ~, exitflag] = lsqnonlin(@(x) pc_scheme(x, dl1, dt, F_old, be_old, me_old, ...
                kappa_old, nu_old, parE, par), x0, lb, ub, options);
        catch ME
            fprintf('Solver exception at step %d: %s\n', i, ME.message);
            success = false;
            return;
        end

        if exitflag <= 0
            success = false;
            return;
        end

        dl2 = x(1);
        dl3 = x(2);
        kappa = x(3);
        nu = x(4:end);

        [~, be_x, me_x] = pc_scheme(x, dl1, dt, F_old, be_old, me_old, kappa_old, nu_old, parE, par);

        dF = [dl1, 0, 0; 0, dl2, 0; 0, 0, dl3];
        F{i} = F{i - 1} + dF;
        b{i} = F{i} * F{i}';
        J{i} = det(F{i});
        be{i} = kappa * (be_x + (1 - 1 / kappa_old) * be{i - 1});
        Je{i} = sqrt(det(be{i}));

        me_iter = cell(parE.NF, 1);
        ae_iter = cell(parE.NF, 1);
        for k = 1:parE.NF
            me{k, i} = nu(k) * (me_x{k} + (1 - 1 / nu_old(k)) * me{k, i - 1});
            ae{k, i} = norm(me{k, i});
            me_iter{k} = me{k, i};
            ae_iter{k} = ae{k, i};
        end

        I_trace = trace(b{i});
        g = comp_g(parE, par, I_trace, J{i}, Je{i}, ae_iter);
        g_save{i} = g;

        sF = zeros(3, 3);
        for k = 1:parE.NF
            sF_k = comp_sF(parE, par, g, ae_iter{k}, me_iter{k}, J{i});
            sF = sF + sF_k / parE.NF;
        end
        sF_save{i} = sF;

        sM = comp_sM(par, g, J{i}, Je{i});
        sM_save{i} = sM;

        sN = comp_sN(par, g, b{i}, J{i});
        sN_save{i} = sN;

        s{i} = sF + sM + sN;
    end

    % Collect vectors
    rTmax_plot = find(~cellfun(@isempty, s), 1, 'last');
    if isempty(rTmax_plot) || rTmax_plot < 2
        success = false;
        return;
    end
    
    s11_raw = zeros(rTmax_plot, 1);
    S11_raw = zeros(rTmax_plot, 1);
    F11_raw = zeros(rTmax_plot, 1);
    F22_raw = zeros(rTmax_plot, 1);
    F33_raw = zeros(rTmax_plot, 1);
    
    for i = 1:rTmax_plot
        s11_raw(i, 1) = s{i}(1, 1);
        F11_raw(i, 1) = F{i}(1, 1);
        F22_raw(i, 1) = F{i}(2, 2);
        F33_raw(i, 1) = F{i}(3, 3);
        % =====================================================================
        % Use computed F11 instead of input lambda1 for S11 calculation
        % =====================================================================
        S11_raw(i, 1) = s11_raw(i, 1) * J{i} / F11_raw(i, 1);
    end
    
    runTime_raw = runTime(1:rTmax_plot);
    lambda1_raw = lambda1(1:rTmax_plot);

    % =====================================================================
    % Downsampling: handle hold phase properly
    % =====================================================================
    has_hold = (scenario == 1) || (hold_time > 0);
    target_dl = 0.001;
    target_dt = 0.1;  % Time step for hold phase interpolation (seconds)
    
    if has_hold
        % =====================================================================
        % Hold detection uses the final lambda value
        % =====================================================================
        lambda_final = lambda1_raw(end);
        % Find first index where lambda reaches final value (within tolerance)
        hold_start_idx = find(abs(lambda1_raw - lambda_final) < 1e-6, 1, 'first');
        
        if isempty(hold_start_idx) || hold_start_idx >= rTmax_plot - 1
            % No actual hold phase found (less than 2 hold points), treat as regular ramp
            has_hold = false;
        else
            % Split into ramp and hold phases
            ramp_idx = 1:hold_start_idx;
            hold_idx = hold_start_idx:rTmax_plot;
            
            % === RAMP PHASE: Interpolate based on lambda ===
            lambda_ramp = lambda1_raw(ramp_idx);
            [lambda_ramp_unique, uniq_idx_ramp] = unique(lambda_ramp, 'stable');
            
            % Need at least 2 unique points for interpolation
            if numel(lambda_ramp_unique) < 2
                has_hold = false;
            else
                s11_ramp = s11_raw(ramp_idx);
                S11_ramp = S11_raw(ramp_idx);
                F11_ramp = F11_raw(ramp_idx);
                F22_ramp = F22_raw(ramp_idx);
                F33_ramp = F33_raw(ramp_idx);
                t_ramp = runTime_raw(ramp_idx);
                
                s11_ramp = s11_ramp(uniq_idx_ramp);
                S11_ramp = S11_ramp(uniq_idx_ramp);
                F11_ramp = F11_ramp(uniq_idx_ramp);
                F22_ramp = F22_ramp(uniq_idx_ramp);
                F33_ramp = F33_ramp(uniq_idx_ramp);
                t_ramp = t_ramp(uniq_idx_ramp);
                
                % Interpolate ramp phase on lambda grid
                lambda_target_ramp = (1.0:target_dl:lambda_ramp_unique(end))';
                if isempty(lambda_target_ramp)
                    lambda_target_ramp = lambda_ramp_unique;
                end
                
                % =====================================================================
                % Use linear interpolation with boundary handling instead of extrap
                % =====================================================================
                s11_interp_ramp = interp1(lambda_ramp_unique, s11_ramp, lambda_target_ramp, 'linear', 'extrap');
                S11_interp_ramp = interp1(lambda_ramp_unique, S11_ramp, lambda_target_ramp, 'linear', 'extrap');
                F11_interp_ramp = interp1(lambda_ramp_unique, F11_ramp, lambda_target_ramp, 'linear', 'extrap');
                F22_interp_ramp = interp1(lambda_ramp_unique, F22_ramp, lambda_target_ramp, 'linear', 'extrap');
                F33_interp_ramp = interp1(lambda_ramp_unique, F33_ramp, lambda_target_ramp, 'linear', 'extrap');
                t_interp_ramp = interp1(lambda_ramp_unique, t_ramp, lambda_target_ramp, 'linear', 'extrap');
                
                % Clamp any negative stresses to zero (physically invalid)
                S11_interp_ramp = max(0, S11_interp_ramp);
                s11_interp_ramp = max(0, s11_interp_ramp);
                
                % === HOLD PHASE: Interpolate based on time ===
                t_hold = runTime_raw(hold_idx);
                s11_hold = s11_raw(hold_idx);
                S11_hold = S11_raw(hold_idx);
                F11_hold = F11_raw(hold_idx);
                F22_hold = F22_raw(hold_idx);
                F33_hold = F33_raw(hold_idx);
                
                t_hold_start = t_hold(1);
                t_hold_end = t_hold(end);
                
                if t_hold_end > t_hold_start + 1e-9 && numel(t_hold) >= 2
                    % Create time grid for hold phase
                    t_target_hold = (t_hold_start:target_dt:t_hold_end)';
                    
                    % Ensure we include the end point
                    if isempty(t_target_hold)
                        t_target_hold = [t_hold_start; t_hold_end];
                    elseif t_target_hold(end) < t_hold_end - 1e-9
                        t_target_hold = [t_target_hold; t_hold_end];
                    end
                    
                    % =====================================================================
                    % Avoid duplication at ramp/hold junction
                    % =====================================================================
                    % Remove first point if it's too close to ramp end
                    if numel(t_target_hold) > 1 && abs(t_target_hold(1) - t_interp_ramp(end)) < target_dt/2
                        t_target_hold = t_target_hold(2:end);
                    end
                    
                    % Check we still have points to interpolate
                    if numel(t_target_hold) >= 1 && numel(t_hold) >= 2
                        % Ensure t_hold is unique for interpolation
                        [t_hold_unique, t_hold_uniq_idx] = unique(t_hold, 'stable');
                        if numel(t_hold_unique) >= 2
                            s11_interp_hold = interp1(t_hold_unique, s11_hold(t_hold_uniq_idx), t_target_hold, 'linear', 'extrap');
                            S11_interp_hold = interp1(t_hold_unique, S11_hold(t_hold_uniq_idx), t_target_hold, 'linear', 'extrap');
                            F11_interp_hold = interp1(t_hold_unique, F11_hold(t_hold_uniq_idx), t_target_hold, 'linear', 'extrap');
                            F22_interp_hold = interp1(t_hold_unique, F22_hold(t_hold_uniq_idx), t_target_hold, 'linear', 'extrap');
                            F33_interp_hold = interp1(t_hold_unique, F33_hold(t_hold_uniq_idx), t_target_hold, 'linear', 'extrap');
                            t_interp_hold = t_target_hold;
                        else
                            % Only one unique time point, use raw values
                            s11_interp_hold = s11_hold(end);
                            S11_interp_hold = S11_hold(end);
                            F11_interp_hold = F11_hold(end);
                            F22_interp_hold = F22_hold(end);
                            F33_interp_hold = F33_hold(end);
                            t_interp_hold = t_hold(end);
                        end
                    else
                        s11_interp_hold = [];
                        S11_interp_hold = [];
                        F11_interp_hold = [];
                        F22_interp_hold = [];
                        F33_interp_hold = [];
                        t_interp_hold = [];
                    end
                else
                    % Hold phase has no duration or not enough points
                    s11_interp_hold = [];
                    S11_interp_hold = [];
                    F11_interp_hold = [];
                    F22_interp_hold = [];
                    F33_interp_hold = [];
                    t_interp_hold = [];
                end
                
                % === CONCATENATE RAMP AND HOLD ===
                s11_final = [s11_interp_ramp(:); s11_interp_hold(:)];
                S11_final = [S11_interp_ramp(:); S11_interp_hold(:)];
                F11_final = [F11_interp_ramp(:); F11_interp_hold(:)];
                F22_final = [F22_interp_ramp(:); F22_interp_hold(:)];
                F33_final = [F33_interp_ramp(:); F33_interp_hold(:)];
                t_final = [t_interp_ramp(:); t_interp_hold(:)];
            end
        end
    end
    
    if ~has_hold
        % === NO HOLD: Original interpolation based on lambda ===
        [lambda_unique, uniq_idx] = unique(lambda1_raw, 'stable');
        
        if numel(lambda_unique) < 2
            % Not enough points for interpolation
            success = false;
            return;
        end
        
        s11_unique = s11_raw(uniq_idx);
        S11_unique = S11_raw(uniq_idx);
        F11_unique = F11_raw(uniq_idx);
        F22_unique = F22_raw(uniq_idx);
        F33_unique = F33_raw(uniq_idx);
        t_unique = runTime_raw(uniq_idx);
        
        lambda_target = (1.0:target_dl:lambda_unique(end))';
        if isempty(lambda_target)
            lambda_target = lambda_unique;
        end
        
        s11_final = interp1(lambda_unique, s11_unique, lambda_target, 'linear', 'extrap');
        S11_final = interp1(lambda_unique, S11_unique, lambda_target, 'linear', 'extrap');
        F11_final = interp1(lambda_unique, F11_unique, lambda_target, 'linear', 'extrap');
        F22_final = interp1(lambda_unique, F22_unique, lambda_target, 'linear', 'extrap');
        F33_final = interp1(lambda_unique, F33_unique, lambda_target, 'linear', 'extrap');
        t_final = interp1(lambda_unique, t_unique, lambda_target, 'linear', 'extrap');
        
        % Clamp negative stresses
        S11_final = max(0, S11_final);
        s11_final = max(0, s11_final);
    end

    % Final output
    curve = struct();
    curve.s11 = s11_final(:);
    curve.S11 = S11_final(:);
    curve.F11 = F11_final(:);
    curve.F22 = F22_final(:);
    curve.F33 = F33_final(:);
    curve.t = t_final(:);
    curve.par = par;
end

function [lambda1, runTime, eff_rate, used_rates] = build_loading_schedule(lambda1_max, lambda1_dot, scenario, n_segments, seg_lambdas, seg_rates, hold_time, solver_dl)
% Build lambda1(t) and time vector for different scenarios.
% Handles all scenarios and ensures lambda1_max is reached.

    target_dl = solver_dl;
    lambda_vals = 1;
    time_vals = 0;
    eff_rate = lambda1_dot;
    used_rates = lambda1_dot;
    
    % Ensure lambda1_dot is positive
    if lambda1_dot <= 0
        lambda1_dot = 0.001;
        eff_rate = lambda1_dot;
    end

    if (scenario == 2) && n_segments > 0
        last_lambda = 1;
        last_time = 0;
        max_segments = min(n_segments, numel(seg_lambdas));
        last_used_rate = lambda1_dot;  % Track last rate used
        
        for idx = 1:max_segments
            seg_target = seg_lambdas(idx);
            seg_rate = seg_rates(idx);
            
            % Validate segment rate
            if isnan(seg_rate) || seg_rate <= 0
                seg_rate = lambda1_dot;
            end
            
            if isnan(seg_target) || seg_target <= last_lambda
                continue;
            end
            
            seg_target = min(seg_target, lambda1_max);
            steps = max(2, ceil((seg_target - last_lambda) / target_dl) + 1);
            lam_seg = linspace(last_lambda, seg_target, steps);
            dt_seg = diff(lam_seg) ./ seg_rate;
            t_seg = last_time + [0; cumsum(dt_seg(:))];
            lambda_vals = [lambda_vals; lam_seg(2:end)'];
            time_vals = [time_vals; t_seg(2:end)];
            last_lambda = seg_target;
            last_time = t_seg(end);
            last_used_rate = seg_rate;
            
            if idx == 1
                eff_rate = seg_rate;
            end
            used_rates = [used_rates, seg_rate];
            
            if seg_target >= lambda1_max
                break;
            end
        end
        
        % =====================================================================
        % Ensure lambda1_max is reached even if segments do not cover it
        % =====================================================================
        if last_lambda < lambda1_max - 1e-9
            % Add final segment to reach lambda1_max using the last rate
            steps = max(2, ceil((lambda1_max - last_lambda) / target_dl) + 1);
            lam_seg = linspace(last_lambda, lambda1_max, steps);
            dt_seg = diff(lam_seg) ./ last_used_rate;
            t_seg = last_time + [0; cumsum(dt_seg(:))];
            lambda_vals = [lambda_vals; lam_seg(2:end)'];
            time_vals = [time_vals; t_seg(2:end)];
        end
    else
        % Simple ramp (scenario 0, 1, or 3 sub-ramps)
        steps = max(2, ceil((lambda1_max - 1) / target_dl) + 1);
        lambda_vals = linspace(1, lambda1_max, steps)';
        time_base = (lambda_vals - 1) / lambda1_dot;
        time_vals = time_base;
        used_rates = lambda1_dot;
    end

    % =====================================================================
    % Hold segment timing calculation
    % =====================================================================
    if (scenario == 1) || (hold_time > 0)
        % Use a reasonable time step for hold phase
        % Based on hold_time, not on rate
        dt_hold_max = 0.1;  % Maximum 0.1 seconds per step
        dt_hold_min = 0.001;  % Minimum 0.001 seconds per step
        
        % Target ~1000 steps for hold, but bounded
        if hold_time > 0
            dt_hold = max(dt_hold_min, min(dt_hold_max, hold_time / 1000));
            n_hold = max(10, ceil(hold_time / dt_hold));  % At least 10 hold steps
        else
            dt_hold = dt_hold_max;
            n_hold = 10;
        end
        
        hold_times = (1:n_hold)' * (hold_time / n_hold) + time_vals(end);
        lambda_vals = [lambda_vals; lambda_vals(end) * ones(n_hold, 1)];
        time_vals = [time_vals; hold_times];
    end

    lambda1 = lambda_vals;
    runTime = time_vals - time_vals(1);
    
    % =====================================================================
    % Ensure time is strictly monotonic
    % =====================================================================
    for ti = 2:numel(runTime)
        if runTime(ti) <= runTime(ti-1)
            runTime(ti) = runTime(ti-1) + 1e-9;
        end
    end
end
