function [DS_params, mapping_gmm] = JTDS_Solver_v3(Data, robotplant, options)
% JTDS_Solver_v3: Task-space LPV-DS + Null-space Preference Mapping (GMM)
% Implements: 
%   1. Learn LPV-DS in task space using GMM
%   2. Learn GMM mapping from task space to preferred joint config

% Inputs:
%   Data: (2*dimq + dimx) x N matrix, as in v2
%   robotplant: robot model
%   options: struct

% Outputs:
%   DS_params: parameters of the learned task-space DS (GMM-based LPV-DS)
%   mapping_gmm: GMM mapping task state to joint config

% Extract the basic parameters of our system
dimq = robotplant.robot.n;
dimx = robotplant.dimx;
n = size(Data, 2); % total number of demonstrated examples

Q = Data(1:dimq, :); % joint positions
Qd = Data(dimq + 1: 2*dimq, :); % joint velocities
Xt = Data(2*dimq + 1:end, :); % target task positions

% Compute task velocities (Xd) from Qd and Q using robotplant
if options.orientation_flag == 1
    task_dim = 6; % Position + orientation
else
    task_dim = 3; % Position only
end

Xd = zeros(task_dim, n);
for i = 1:n
    q = Q(:, i);
    J = robotplant.robot.jacob0(q);
    if options.orientation_flag == 1
        J_task = J(1:6, :);
    else
        J_task = J(1:3, :);
    end
    Xd(:, i) = J_task * Qd(:, i);
end

% --- Step 1: Learn LPV-DS in Task Space using ch3_ex4_lpvDS approach ---
task_dim = size(Xt, 1);

% Set default options for LPV-DS estimation
if ~isfield(options, 'lpv_est_options')
    est_options = [];
    est_options.type             = 0;   % GMM Estimation Algorithm Type 
    est_options.maxK             = 10;  % Maximum Gaussians for Type 1
    est_options.samplerIter      = 30;  % Maximum Sampler Iterations
    est_options.do_plots         = 0;   % Do not plot estimation statistics
    est_options.sub_sample       = 1;   % Sub-sampling factor
    est_options.estimate_l       = 1;   % Estimate lengthscale
    est_options.l_sensitivity    = 2;   % Lengthscale sensitivity
    est_options.length_scale     = []; 
else
    est_options = options.lpv_est_options;
end

if options.verbose
    disp('Step 1: Fitting GMM to task space data...');
end

% Fit GMM to task space trajectory data (position + velocity)
[Priors_task, Mu_task, Sigma_task] = fit_gmm(Xt, Xd, est_options);

% Create ds_gmm structure
ds_gmm_task = [];
ds_gmm_task.Priors = Priors_task;
ds_gmm_task.Mu = Mu_task;
ds_gmm_task.Sigma = Sigma_task;

% (Optional) Adjust covariances for smoother dynamics
if ~isfield(options, 'adjust_covariances')
    options.adjust_covariances = false;
end

if options.adjust_covariances
    if task_dim == 2
        tot_dilation_factor = 1; rel_dilation_fact = 0.2;
    elseif task_dim == 3
        tot_dilation_factor = 1; rel_dilation_fact = 0.5;
    elseif task_dim == 6
        tot_dilation_factor = 1; rel_dilation_fact = 0.75;
    else
        tot_dilation_factor = 1; rel_dilation_fact = 0.5;
    end
    Sigma_adjusted = adjust_Covariances(ds_gmm_task.Priors, ds_gmm_task.Sigma, tot_dilation_factor, rel_dilation_fact);
    ds_gmm_task.Sigma = Sigma_adjusted;
end

if options.verbose
    disp('Step 1: Learning LPV-DS dynamics matrices...');
end

% Set up LPV-DS optimization options
if ~isfield(options, 'constr_type')
    options.constr_type = 2;  % 0:'convex', 1:'non-convex', 2:'non-convex with P'
end
if ~isfield(options, 'init_cvx')
    options.init_cvx = 1;
end

constr_type = options.constr_type;
init_cvx = options.init_cvx;

% Define attractor in task space (assumed to be target position)
if size(Xt, 2) > 0
    att_task = Xt(:, end); % Use last target position as attractor
else
    att_task = zeros(task_dim, 1); % Default to origin
end

% P-matrix learning for constraint type 2
if constr_type == 0 || constr_type == 1
    P_opt = eye(task_dim);
else
    % Prepare data for P-matrix learning (shift to attractor)
    TaskData_shifted = [Xt - repmat(att_task, 1, n); Xd];
    [Vxf] = learn_wsaqf(TaskData_shifted);
    P_opt = Vxf.P;
    if options.verbose
        fprintf('P matrix pre-estimated for task space.\n');
    end
end

% Learn LPV-DS dynamics matrices
TaskData_full = [Xt; Xd]; % Combine position and velocity
if constr_type == 1
    [A_k_task, b_k_task, P_est_task] = optimize_lpv_ds_from_data(TaskData_full, zeros(task_dim,1), constr_type, ds_gmm_task, P_opt, init_cvx);
    ds_task = @(x) lpv_ds(x-repmat(att_task, [1 size(x,2)]), ds_gmm_task, A_k_task, b_k_task);
else
    [A_k_task, b_k_task, ~] = optimize_lpv_ds_from_data(TaskData_full, att_task, constr_type, ds_gmm_task, P_opt, init_cvx);
    ds_task = @(x) lpv_ds(x, ds_gmm_task, A_k_task, b_k_task);
end

% Store DS parameters
DS_params = [];
DS_params.GMM = ds_gmm_task;
DS_params.A_k = A_k_task;
DS_params.b_k = b_k_task;
DS_params.att = att_task;
DS_params.task_dim = task_dim;
DS_params.ds_fun = ds_task;
DS_params.constr_type = constr_type;
if constr_type == 1
    DS_params.P_est = P_est_task;
else
    DS_params.P_opt = P_opt;
end

if options.verbose
    fprintf('Task space LPV-DS: Fitted with %d GMM components\n', length(Priors_task));
end

% --- Step 2: Learn Joint GMM p(x, q_null) ---
if options.verbose
    disp('Step 2: Extracting null space components...');
end

% Extract null space components from demonstrations
Q_null = zeros(dimq, n);
for i = 1:n
    q = Q(:, i);
    x_task = Xt(:, i);
    
    % Compute Jacobian and null space projector
    J = robotplant.robot.jacob0(q);
    J_task = J(1:task_dim, :);
    J_pinv = pinv(J_task);
    N = eye(dimq) - J_pinv * J_task;
    
    Q_null(:, i) = N * q;
    
end

% Set default options for null space mapping GMM
if ~isfield(options, 'mapping_GMM_sigma_type')
    options.mapping_GMM_sigma_type = 'full';
end
if ~isfield(options, 'mapping_max_gaussians')
    options.mapping_max_gaussians = 8;
end
if ~isfield(options, 'mapping_GMM_num_replicates')
    options.mapping_GMM_num_replicates = 20;
end

% Learn joint distribution p(x, q_null)
if options.verbose
    disp('Step 2: Learning joint GMM p(x, q_null)...');
end

JointNullData = [Xt' Q_null'];  % [task_positions, null_space_configs]
% Find optimal number of Gaussians for null space mapping
best_BIC_mapping = inf;
best_k_mapping = 1;

for k_tmp = 1:options.mapping_max_gaussians
    warning('off', 'all');
    GMM_tmp = fitgmdist(JointNullData, k_tmp, 'Start', 'plus', ...
        'CovarianceType', options.mapping_GMM_sigma_type, ...
        'Regularize', 0.000001, 'Replicates', options.mapping_GMM_num_replicates);
    warning('on', 'all');
    
    % Compute BIC
    log_likelihood = sum(log(pdf(GMM_tmp, JointNullData)));
    num_params = k_tmp * ((task_dim + dimq) + (task_dim + dimq)*((task_dim + dimq)+1)/2);
    bic_tmp = -2*log_likelihood + num_params*log(n);
    
    if bic_tmp < best_BIC_mapping
        best_BIC_mapping = bic_tmp;
        best_k_mapping = k_tmp;
    end
end

% Fit final mapping GMM
warning('off', 'all');
gmm_fit = fitgmdist(JointNullData, best_k_mapping, 'Start', 'plus', ...
    'CovarianceType', options.mapping_GMM_sigma_type, ...
    'Regularize', 0.000001, 'Replicates', options.mapping_GMM_num_replicates);
warning('on', 'all');

% Create custom structure to store GMM and additional parameters
mapping_gmm = [];
mapping_gmm.gmm = gmm_fit;  % Store the actual GMM object
mapping_gmm.task_dim = task_dim;
mapping_gmm.joint_dim = dimq;
mapping_gmm.num_components = best_k_mapping;

if options.verbose
    fprintf('Null space mapping: Fitted GMM with %d components\n', best_k_mapping);
end

% --- Step 3: Execution Policy (for reference, implement in demo script) ---
% At runtime, for current q and phi (task state):
%   1. phi_dot = eval_LPV_DS(DS_params, phi)
%   2. J = robotplant.robot.jacob0(q); J_pinv = pinv(J_task);
%   3. N = eye(dimq) - J_pinv * J_task;
%   4. q_null = sample or mean from mapping_gmm.gmm given phi
%   5. q_dot = J_pinv * phi_dot + N * k * (q_null - q);
%
% Example usage for regression:
%   query_point = [x_current; zeros(dimq, 1)];  % Task position + dummy null space
%   [posterior_means, posterior_covs] = posterior(mapping_gmm.gmm, query_point);
%   q_null_desired = mean of conditional distribution p(q_null | x_current)

end
