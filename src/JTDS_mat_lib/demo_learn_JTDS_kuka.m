%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Learning JTDS Models (including orientation) on Different Datasets     %%
%%     NEW: Joint+Task Space Multi-Objective Learning with Multiple Approaches%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This demo supports multiple learning approaches for JTDS:
%
% OPTIONS TO SET (in STEP 2):
%
% 1. APPROACH SELECTION:
%    - options.learn_task_space: Enable task space learning (true/false)
%    - options.use_null_space: Use null space projection (true) or weighted sum (false)
%    - options.task_space_weight: Weight for task space (0.0 to 1.0, only for weighted sum)
%    - options.null_space_mapping: Learn mapping from task space to null space (true/false)
%
% 2. COMPARISON MODE:
%    - run_comparison: Compare all approaches (true) or run single approach
%    (false) [NOT FULLY WORKING! 4TH APPROACH IS NOT INCLUDED]
%
% AVAILABLE APPROACHES:
%
% 1. Joint Space Only:
%    - learn_task_space = false
%    - Learns only joint space dynamics, converging to the task space goal
%    - Original JT-DS approach
%
% 2. Weighted Sum (Multi-objective):
%    - learn_task_space = true, use_null_space = false
%    - Minimizes: (1-w)*J_joint + w*J_task
%
% 3. Null Space Projection:
%    - learn_task_space = true, use_null_space = true, null_space_mapping = false
%    - Task space primary, joint space projected to null space
%
% 4. Null space mapping:
%    - learn_task_space = true, use_null_space = true, null_space_mapping = true
%    - Learns mapping from task space to null space, then projects
%
% USAGE EXAMPLES:
% - For presentation comparison: Set run_comparison = true
% - For specific approach: Set run_comparison = false, configure options above
% - For task accuracy: Use approach 3 (null space projection)
% - For joint smoothness: Use approach 1 (joint space only) or 2 with low weight


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 1: Load and Process dataset %
%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

% Set random seed for reproducible results
% rng(12345, 'twister');  % Use a fixed seed
% fprintf('Random seed set to 12345 for reproducible results\n');

do_plots  = 1;
data_path = '../../Data/mat/'; % <-Insert path to datasets folder here
choosen_dataset = 'back'; % Options: 'back','fore','pour','pour_obst','foot','singularity';

switch choosen_dataset
    case 'back'
        demos_location = strcat(data_path, 'back_hand/data.mat');
        demo_ids = [2:11];
    case 'fore'
        demos_location = strcat(data_path,'fore_hand/data.mat');
        demo_ids = [1:11];
    case 'pour' 
        demos_location = strcat(data_path,'pour_no_obst/data.mat');
        demo_ids = [1 2 3 5 6 7 8 9 10];
    case 'pour_obst'
        demos_location = strcat(data_path,'pour_obst/data.mat');
        demo_ids = [1:10];
    case 'pour_obst_2'
        demos_location = strcat(data_path,'pour_obst_2/data.mat');
        demo_ids = [1:7];                
    case 'foot'        % This dataset was recorded at 50 Hz! thinning_ratio = 1 or 2
        demos_location = strcat(data_path,'foot/data.mat');
        demo_ids = [1:8];                
    case 'singularity'   
        demos_location = strcat(data_path,'singularity/data.mat');
        demo_ids = [1:10];  
        fprintf('Loading demonstrations from %s \n', demos_location);
        load(demos_location)
end

if ~strcmp(choosen_dataset,'singularity')
    fprintf('Loading demonstrations from %s \n', demos_location);
    [Qs_, Ts_] = ImportDemonstrations(demos_location);
end
% If the data is very dense, initializing the semidefinite program may take
% a long time. In this case, it may help to thin down the number of
% demonstrated points (by varying "thinning_ratio", so long as there are still sufficient points to
% satisfactorily reconstruct the shape of the trajectory.
% In the KUKA case, we get 500 datapoints per second, so we recommend shrinking the data density considerably
thinning_ratio = 20; % Same as demonstrations recorded at 10->50Hz, 20->25Hz
Qs = []; Ts= [];
for i = 1:length(demo_ids)
    Qs{i,1} = Qs_{demo_ids(i)}(:, 1:thinning_ratio:end);
    Ts{i,1} = Ts_{demo_ids(i)}(:, 1:thinning_ratio:end);
end

if do_plots
    figure('Color',[1 1 1])    
    for i=1:length(Qs)
        Data_ = [];
        Data_ = Qs{i};
        subplot(1,2,1)
        scatter3(Data_(1,:),Data_(2,:),Data_(3,:),10,'filled'); hold on;
        xlabel('$q_1$','Interpreter','LaTex');ylabel('$q_2$','Interpreter','LaTex');zlabel('$q_3$','Interpreter','LaTex')
        title('First 3 Joint Angles (Raw)', 'Interpreter','LaTex')
        subplot(1,2,2)
        scatter3(Data_(4,:),Data_(5,:),Data_(6,:),10,'filled'); hold on;
        title('Last 3 Joint Angles (Raw)', 'Interpreter','LaTex')
        xlabel('$q_4$','Interpreter','LaTex');ylabel('$q_5$','Interpreter','LaTex');zlabel('$q_6$','Interpreter','LaTex')
    end    
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  STEP 2: Prepare Data for Learning and Initial Model Parameters   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Choose Lower-Dimensional Mapping Technique
mapping = {'None'}; % 'None', 'PCA', 'KPCA'

%%% Learning options %%%
options = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To remove orientation from target 
% simply set flag = 0 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.orientation_flag = 0; 
options.tol_cutting = 0.1;

%%% Dim-Red options %%%
options.explained_variance_threshold = .95;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If choosen mapping is K-PCA you 
% need to choose the kernel width
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% options.kpca_sigma = mean_D/sqrt(2);

%%% GMM options %%%
options.GMM_sigma_type = 'full'; % Can be either 'full' or 'diagonal'
options.GMM_maximize_BIC = true;
options.max_gaussians = 10;
options.plot_BIC = 1; 

% Optimization options 
options.learn_with_bounds = false;
options.verbose = true;

% Task space learning options
options.learn_task_space = true;        % Enable joint+task space learning
options.task_space_weight = 0.8;        % weight for learning task space 
options.use_null_space = true;         % Use weighted sum (false) or null space projection (true) 
options.null_space_mapping = false;     % Use null space mapping (Approach 4) - triggers JTDS_Solver_v3

% Comparison options
run_comparison = false;                  % Set to true to compare all approaches, false to run only chosen approach

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DH parameters for the KUKA LWR 4+ robot %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dimq = 7;
A = [0 0 0 0 0 0 0.05];
Alpha = pi/2*[1 -1 -1 1 1 -1 0];
D = [.34 0 .4 0 .4 0 .279];
Qmin = 2*pi/180*[-85, -90, -100, -110, -140, -90, -120];
Qmax = 2*pi/180*[85, 90, 100, 110, 140, 90, 120];
% Create a model of the robot
robot = initialize_robot(A,D,Alpha,Qmin,Qmax);    
% Create a model plant for the robot's motor controller
robotplant = RobotPlant(robot, 'end_trans');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Split Dataset for Training/Testing        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt_ratio = 0.6;
train = round(length(Qs)*tt_ratio);
Qs_train = []; Ts_train = [];
Qs_test = [];   Ts_test = [];
rand_ids = [1:length(Qs)];
for ii=1:length(Qs)
    if ii < train
        Qs_train{ii,1} = Qs{rand_ids(ii)}; Ts_train{ii,1} = Ts{rand_ids(ii)};
    else
        Qs_test{ii-train+1,1} = Qs{rand_ids(ii)}; Ts_test{ii-train+1,1} = Ts{rand_ids(ii)};
    end
end

%%% Pre-process Data %%%
[Data_train, index_train] = preprocess_demos_jtds(robotplant, Qs_train, Ts_train, options.tol_cutting,options.orientation_flag);
[Data_test, index_test]  = preprocess_demos_jtds(robotplant, Qs_test, Ts_test, options.tol_cutting,options.orientation_flag);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    STEP 3:   Learn JTDS model for the Current Dataset      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mapping_name = mapping{1};
fprintf('Training JTDS generator using %s mapping...\n', mapping_name);
options.latent_mapping_type = mapping_name;

% Initialize variables
DS_params = [];
mapping_gmm = [];
using_v3 = false;

% Check which approach to use based on options
if options.learn_task_space && options.use_null_space && isfield(options, 'null_space_mapping') && options.null_space_mapping
    % Approach 4: Null space mapping - Use JTDS_Solver_v3
    fprintf('Using Approach 4: Null space mapping (JTDS_Solver_v3)\n');
    using_v3 = true;
    
    % Start timing
    tic;
    % Run JTDS Solver v3 - Task Space LPV-DS + Null Space Mapping
    [DS_params, mapping_gmm] = JTDS_Solver_v3(Data_train, robotplant, options);
    training_time = toc;
    
    % Create specialized motion generator for v3 approach
    v3_options = struct();
    v3_options.null_gain = 1.0;  % Null space attraction gain
    v3_options.orientation_flag = options.orientation_flag;
    
    motion_generator_v3 = MotionGeneratorV3(robotplant, DS_params, mapping_gmm, v3_options);
    
    % For compatibility with rest of code, create dummy parameters
    % (these won't be used for RMSE computation but needed for some plotting functions)
    K = length(DS_params.GMM.Priors);
    Priors = DS_params.GMM.Priors;
    Mu = zeros(dimq, K);  % Dummy joint space means  
    Sigma = zeros(dimq, dimq, K);  % Dummy joint space covariances
    As = zeros(dimq, dimq, K);  % Dummy dynamics matrices
    
    % Create dummy latent_mapping
    latent_mapping = [];
    latent_mapping.name = 'TaskSpace_v3';
    latent_mapping.M = eye(dimq);
    latent_mapping.mean = zeros(dimq, 1);
    
    fprintf('JTDS_Solver_v3 completed successfully!\n');
    fprintf('Training time: %.2f seconds\n', training_time);
    fprintf('Task space LPV-DS learned with %d GMM components\n', K);
    fprintf('Null space mapping learned with %d GMM components\n', mapping_gmm.num_components);
    
    % Debug: Check if attractor matches target
    fprintf('\n=== ATTRACTOR DEBUGGING ===\n');
    fprintf('Learned attractor: [%.4f, %.4f, %.4f]\n', DS_params.att(1), DS_params.att(2), DS_params.att(3));
    
    % Check what the target should be
    target_demo_pos = Data_train(2*dimq+1:2*dimq+3, end);  % Last point of demonstration
    fprintf('Demo target position: [%.4f, %.4f, %.4f]\n', target_demo_pos(1), target_demo_pos(2), target_demo_pos(3));
    fprintf('Attractor vs demo target distance: %.6f m\n', norm(DS_params.att - target_demo_pos));
    
    % Test DS at attractor
    vel_at_attractor_debug = DS_params.ds_fun(DS_params.att);
    fprintf('Velocity at attractor: [%.6f, %.6f, %.6f], norm: %.6f\n', ...
        vel_at_attractor_debug(1), vel_at_attractor_debug(2), vel_at_attractor_debug(3), norm(vel_at_attractor_debug));
    
    % Test DS at a few points along the trajectory
    fprintf('Testing DS at various points:\n');
    for test_i = [1, 25, 50, 75, 100]
        if test_i <= size(Data_train, 2)
            test_pos = Data_train(2*dimq+1:2*dimq+3, test_i);
            test_vel = DS_params.ds_fun(test_pos);
            fprintf('  Point %d: pos=[%.3f,%.3f,%.3f], DS_vel_norm=%.4f\n', ...
                test_i, test_pos(1), test_pos(2), test_pos(3), norm(test_vel));
        end
    end
    
    % Check eigenvalues of dynamics matrices (should be negative for stability)
    fprintf('\n=== STABILITY ANALYSIS ===\n');
    all_stable = true;
    for k = 1:length(DS_params.GMM.Priors)
        eigenvals = eig(DS_params.A_k(:,:,k));
        fprintf('  Component %d: [%.3f, %.3f, %.3f] (should be negative)\n', k, eigenvals(1), eigenvals(2), eigenvals(3));
        if any(real(eigenvals) > -1e-6)
            fprintf('    ⚠️  Component %d has unstable eigenvalues!\n', k);
            all_stable = false;
        end
    end
    
    if all_stable
        fprintf('  ✓ All components are stable\n');
    else
        fprintf('  ✗ Some components are unstable - this will cause divergence!\n');
    end
    
    % Check GMM quality
    fprintf('\n=== GMM QUALITY CHECK ===\n');
    fprintf('GMM Prior weights:\n');
    for k = 1:length(DS_params.GMM.Priors)
        fprintf('  Component %d: %.3f\n', k, DS_params.GMM.Priors(k));
    end
    
else
    % All other approaches - Use JTDS_Solver_v2
    if ~options.learn_task_space
        fprintf('Using Approach 1: Joint Space Only (JTDS_Solver_v2)\n');
    elseif options.learn_task_space && ~options.use_null_space
        fprintf('Using Approach 2: Weighted Sum (JTDS_Solver_v2)\n');
    elseif options.learn_task_space && options.use_null_space
        fprintf('Using Approach 3: Null Space Projection (JTDS_Solver_v2)\n');
    end
    
    % Start timing
    tic;
    % Run JTDS Solver v2
    [Priors, Mu, Sigma, As, latent_mapping] = JTDS_Solver_v2(Data_train, robotplant, options);
    training_time = toc;
    
    K = length(Priors);
    fprintf('JTDS_Solver_v2 completed successfully!\n');
    fprintf('Training time: %.2f seconds\n', training_time);
end

if strcmp('PCA',latent_mapping.name)
    pca_dim = length(latent_mapping.lambda);
end

% Select the appropriate motion generator based on approach
if using_v3
    motion_generator = motion_generator_v3;  % Use specialized v3 motion generator
    fprintf('Created MotionGeneratorV3 for task-space LPV-DS with null space mapping\n');
else
    % Generate Trajectories from Learnt JTDS - Use original MotionGenerator for v2
    motion_generator = MotionGenerator(robotplant, Mu, Sigma, Priors, As, latent_mapping);
    fprintf('Created MotionGenerator with v2 joint-space parameters\n');
end

% Compute RMSE on training data
rmse_train_joint = mean(trajectory_error(motion_generator, Data_train(1:dimq, :), Data_train(dimq+1:2*dimq, :), Data_train(2*dimq+1:end, :),options.orientation_flag))
rmse_train_task = compute_task_space_rmse(motion_generator, Data_train(1:dimq, :), Data_train(dimq+1:2*dimq, :), Data_train(2*dimq+1:end, :), robotplant, options.orientation_flag);

% Compute RMSE on testing data  
rmse_test_joint = mean(trajectory_error(motion_generator, Data_test(1:dimq, :), Data_test(dimq+1:2*dimq, :), Data_test(2*dimq+1:end, :),options.orientation_flag))
rmse_test_task = compute_task_space_rmse(motion_generator, Data_test(1:dimq, :), Data_test(dimq+1:2*dimq, :), Data_test(2*dimq+1:end, :), robotplant, options.orientation_flag);

fprintf('\n=== PERFORMANCE METRICS ===\n');
if using_v3
    fprintf('JTDS v3 (Task Space LPV-DS + Null Space Mapping):\n');
else
    fprintf('%s approach:\n', mapping_name);
end
fprintf('  Joint Space RMSE:\n');
fprintf('    Training: %.6f rad/s\n', rmse_train_joint);
fprintf('    Testing:  %.6f rad/s\n', rmse_test_joint);
fprintf('  Task Space RMSE:\n');
fprintf('    Training: %.6f m/s\n', rmse_train_task);
fprintf('    Testing:  %.6f m/s\n', rmse_test_task);

% For backward compatibility
rmse_train = rmse_train_joint;
rmse_test = rmse_test_joint;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    STEP 3.5: Compare Different Learning Approaches         %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if run_comparison
    fprintf('\n=== RUNNING COMPARISON OF ALL APPROACHES ===\n');

    % 1. Joint-space-only model
    fprintf('\n=== Training joint-space-only model ===\n');
    options_joint_only = options;
    options_joint_only.learn_task_space = false;
    tic;
    [Priors_joint, Mu_joint, Sigma_joint, As_joint, latent_mapping_joint] = JTDS_Solver_v2(Data_train, robotplant, options_joint_only);
    joint_training_time = toc;
    fprintf('Joint-space training time: %.2f seconds\n', joint_training_time);
    motion_generator_joint = MotionGenerator(robotplant, Mu_joint, Sigma_joint, Priors_joint, As_joint, latent_mapping_joint);

    % 2. Weighted sum approach (current multi-objective)
    fprintf('\n=== Training weighted sum model ===\n');
    options_weighted = options;
    options_weighted.use_null_space = false;
    tic;
    [Priors_weighted, Mu_weighted, Sigma_weighted, As_weighted, latent_mapping_weighted] = JTDS_Solver_v2(Data_train, robotplant, options_weighted);
    weighted_training_time = toc;
    fprintf('Weighted sum training time: %.2f seconds\n', weighted_training_time);
    motion_generator_weighted = MotionGenerator(robotplant, Mu_weighted, Sigma_weighted, Priors_weighted, As_weighted, latent_mapping_weighted);

    % 3. Null space projection approach
    fprintf('\n=== Training null space projection model ===\n');
    options_nullspace = options;
    options_nullspace.use_null_space = true;
    options_nullspace.null_space_mapping = false;  % Use the stable approach
    tic;
    [Priors_nullspace, Mu_nullspace, Sigma_nullspace, As_nullspace, latent_mapping_nullspace] = JTDS_Solver_v2(Data_train, robotplant, options_nullspace);
    nullspace_training_time = toc;
    fprintf('Null space projection training time: %.2f seconds\n', nullspace_training_time);
    motion_generator_nullspace = MotionGenerator(robotplant, Mu_nullspace, Sigma_nullspace, Priors_nullspace, As_nullspace, latent_mapping_nullspace);    % Compare RMSE on training data
    rmse_train_joint_only = mean(trajectory_error(motion_generator_joint, Data_train(1:dimq, :), Data_train(dimq+1:2*dimq, :), Data_train(2*dimq+1:end, :),options.orientation_flag));
    rmse_train_weighted_sum = mean(trajectory_error(motion_generator_weighted, Data_train(1:dimq, :), Data_train(dimq+1:2*dimq, :), Data_train(2*dimq+1:end, :),options.orientation_flag));
    rmse_train_nullspace_proj = mean(trajectory_error(motion_generator_nullspace, Data_train(1:dimq, :), Data_train(dimq+1:2*dimq, :), Data_train(2*dimq+1:end, :),options.orientation_flag));

    % Compare RMSE on testing data
    rmse_test_joint_only = mean(trajectory_error(motion_generator_joint, Data_test(1:dimq, :), Data_test(dimq+1:2*dimq, :), Data_test(2*dimq+1:end, :),options.orientation_flag));
    rmse_test_weighted_sum = mean(trajectory_error(motion_generator_weighted, Data_test(1:dimq, :), Data_test(dimq+1:2*dimq, :), Data_test(2*dimq+1:end, :),options.orientation_flag));
    rmse_test_nullspace_proj = mean(trajectory_error(motion_generator_nullspace, Data_test(1:dimq, :), Data_test(dimq+1:2*dimq, :), Data_test(2*dimq+1:end, :),options.orientation_flag));

    fprintf('\n=== COMPARISON RESULTS ===\n');
    fprintf('Joint Space Only:\n');
    fprintf('  Training RMSE: %.6f\n', rmse_train_joint_only);
    fprintf('  Testing RMSE:  %.6f\n', rmse_test_joint_only);
    fprintf('Weighted Sum (w=%.1f):\n', options.task_space_weight);
    fprintf('  Training RMSE: %.6f\n', rmse_train_weighted_sum);
    fprintf('  Testing RMSE:  %.6f\n', rmse_test_weighted_sum);
    fprintf('Null Space Projection:\n');
    fprintf('  Training RMSE: %.6f\n', rmse_train_nullspace_proj);
    fprintf('  Testing RMSE:  %.6f\n', rmse_test_nullspace_proj);

    % Use the null space model as the main model for further analysis
    motion_generator = motion_generator_nullspace;
    Priors = Priors_nullspace;
    Mu = Mu_nullspace;
    Sigma = Sigma_nullspace;
    As = As_nullspace;
    latent_mapping = latent_mapping_nullspace;

    % Store all generators for later plotting
    motion_generator_comparison = {motion_generator_joint, motion_generator_weighted, motion_generator_nullspace};
    approach_names = {'Joint Only', 'Weighted Sum', 'Null Space'};

else
    fprintf('\n=== RUNNING SINGLE APPROACH ===\n');
    % Determine which approach is being used
    if ~options.learn_task_space
        fprintf('Using Approach 1: Joint Space Only\n');
        approach_name = 'Joint Space Only';
    elseif options.learn_task_space && ~options.use_null_space
        fprintf('Using Approach 2: Weighted Sum (w=%.1f)\n', options.task_space_weight);
        approach_name = sprintf('Weighted Sum (w=%.1f)', options.task_space_weight);    elseif options.learn_task_space && options.use_null_space
        if isfield(options, 'null_space_mapping') && options.null_space_mapping
            fprintf('Using Approach 4: Null space mapping\n');
            approach_name = 'Null Space Mapping (v3)';
        else
            fprintf('Using Approach 3: Null Space Projection\n');
            approach_name = 'Null Space Projection';
        end
    end

    fprintf('Single approach training completed. RMSE: %.6f (train), %.6f (test)\n', rmse_train, rmse_test);

    % For plotting consistency, create single-element arrays
    motion_generator_comparison = {motion_generator};
    approach_names = {approach_name};
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    STEP 4:  Plot Lower-Dimensional Embedding and Synergies   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp('PCA',latent_mapping.name)
% Extract Lower Dimensional Embedding of Demonstrations
figure('Color',[1 1 1])  
pca_dim  = size(latent_mapping.M,2);
for p=1:pca_dim
    subplot(pca_dim,1,p)    
    for i=1:length(Qs)
        q_ref = Qs{i};
        phi_q = out_of_sample(q_ref', latent_mapping)';
        plot(phi_q(p,:),'-.'); hold on;
    end      
    grid on;
    title(sprintf('Raw Trajectories in PCA-space $\\phi_%d(q)$',p), 'Interpreter', 'LaTex', 'Fontsize', 15)
    xlabel('Time (samples)','Interpreter', 'LaTex', 'Fontsize', 15)   
end

% Plot in 3D-space
if pca_dim == 3
    K_colors = hsv(K);
    figure('Color',[1 1 1])
    for i=1:length(Qs)
        q_ref = Qs{i};
        phi_q = out_of_sample(q_ref', latent_mapping)';
        % Hard clustering for each local model
        labels =  my_gmm_cluster(phi_q, Priors, Mu, Sigma, 'hard', []);
        for k=1:K
            phi_q_k   = phi_q(:,labels==k);
            scatter3(phi_q_k(1,:),phi_q_k(2,:),phi_q_k(3,:),20,K_colors(k,:),'*'); hold on;
        end
        scatter3(phi_q(1,end),phi_q(2,end),phi_q(3,end),100,'o','filled','MarkerEdgeColor','k','MarkerFaceColor',[0 0 0]); hold on;
        axis tight
    end
    
    % Plot Gaussians for local behavior cluster --> might change this to
    % coloring the data-point with the posterior probability
    handles = my_plot3dGaussian(Priors, Mu, Sigma );
    grid on;
    title('Clustered (GMM) Trajectories in PCA-space $\phi(q)$', 'Interpreter', 'LaTex', 'Fontsize', 15)
    xlabel('$\phi_1(q)$', 'Interpreter', 'LaTex', 'Fontsize', 15)
    ylabel('$\phi_2(q)$', 'Interpreter', 'LaTex', 'Fontsize', 15)
    zlabel('$\phi_3(q)$', 'Interpreter', 'LaTex', 'Fontsize', 15)
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   STEP 5: If you are happy with the results, export the model       %%
%%        for execution with the rtk-DS cpp class                      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_dir = strcat('./learned_JTDS_models/',choosen_dataset);
mkdir(model_dir); 
cd(model_dir)
if strcmp(mapping_name, 'None')
    M_p = eye(7);
else
    M_p = latent_mapping.M';
end
out = export2JSEDS_Cpp_lib_v2(Priors, Mu, Sigma, As, latent_mapping.M', latent_mapping.mean, Data_train, index_train);

% save mat file of variables
M = latent_mapping.M';
save('model.mat','Priors','Mu','Sigma', 'As', 'M', 'Data_train', 'index_train','robotplant','latent_mapping')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                    THE FOLLOWING CODE BLOCKS ARE ONLY FOR DEBUGGING!!!                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reproduce learned motion and compute tasks-space metrics == Works with JTDS on position only! %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Extract task-space demonstrations %%%
selected_demo = 1;
num = 100;
task_space_traj = zeros(9,num);

% Generate Task-Space Trajectories
for i=1:num
    trans_tmp=robotplant.robot.fkine(Data_train(1:7,i));
    task_space_traj(:,i) = [trans_tmp(1:3,1); trans_tmp(1:3,2); trans_tmp(1:3,end)];
end

if options.orientation_flag
    % orientation + 3D position target
    x_target = task_space_traj(:,end);
else
    % 3D position target
    x_target = task_space_traj(end-2:end,end);
end


% Plot reproduced trajectory vs. demonstration in joint space
figure('Color',[1 1 1])
q_init         = Data_train(1:dimq,1);

% Adjust parameters for v3 approach
if using_v3
    goal_tolerance = 0.1;
    max_trajectory_duration = 100;  % Set max duration to prevent infinite loops
    fprintf('Using simple trajectory generation for v3 (avoiding ODE integration issues)\n');
    
    % Use simple fixed-step trajectory generation for v3
    dt = 0.005;  % Reduce time step from 0.01 to 0.005 for better accuracy
    
    % Debug: Test DS before trajectory generation
    fprintf('\n=== PRE-TRAJECTORY DS TEST ===\n');
    initial_task_pos = robotplant.robot.fkine(q_init');
    initial_task_pos = initial_task_pos(1:3, 4);
    initial_ds_vel = motion_generator.eval_task_ds(initial_task_pos);
    fprintf('Initial position: [%.4f, %.4f, %.4f]\n', initial_task_pos(1), initial_task_pos(2), initial_task_pos(3));
    fprintf('Initial DS velocity: [%.4f, %.4f, %.4f], norm: %.4f\n', ...
        initial_ds_vel(1), initial_ds_vel(2), initial_ds_vel(3), norm(initial_ds_vel));
    fprintf('Direction to target: [%.4f, %.4f, %.4f]\n', ...
        (x_target - initial_task_pos) / norm(x_target - initial_task_pos));
    
    [Q_traj_JTDS, phi_traj_v3, T_traj_JTDS] = motion_generator.generate_simple_trajectory(q_init, dt, max_trajectory_duration, x_target);
    
    % --- DEBUGGING: Trajectory Generation Analysis ---
    fprintf('\n=== TRAJECTORY GENERATION DEBUGGING ===\n');
    fprintf('Initial task position: [%.4f, %.4f, %.4f]\n', phi_traj_v3(1,1), phi_traj_v3(2,1), phi_traj_v3(3,1));
    fprintf('Final task position: [%.4f, %.4f, %.4f]\n', phi_traj_v3(1,end), phi_traj_v3(2,end), phi_traj_v3(3,end));
    fprintf('Target task position: [%.4f, %.4f, %.4f]\n', x_target(1), x_target(2), x_target(3));
    fprintf('Distance to target: %.6f m\n', norm(phi_traj_v3(:,end) - x_target));
    
    % Check task space trajectory smoothness
    phi_diff = diff(phi_traj_v3, 1, 2);
    phi_velocities = phi_diff / dt;
    max_phi_vel = max(sqrt(sum(phi_velocities.^2, 1)));
    mean_phi_vel = mean(sqrt(sum(phi_velocities.^2, 1)));
    fprintf('Max task velocity: %.4f m/s\n', max_phi_vel);
    fprintf('Mean task velocity: %.4f m/s\n', mean_phi_vel);
    
    % Check for any sudden jumps in task space
    phi_accel = diff(phi_velocities, 1, 2) / dt;
    max_phi_accel = max(sqrt(sum(phi_accel.^2, 1)));
    fprintf('Max task acceleration: %.4f m/s^2\n', max_phi_accel);
    
    % Convert to format expected by plotting functions
    Q_traj_JTDS = Q_traj_JTDS';  % Transpose for compatibility
    T_traj_JTDS = T_traj_JTDS';
    
    fprintf('Generated v3 trajectory with %d points over %.2f seconds\n', size(Q_traj_JTDS, 1), T_traj_JTDS(end));
else
    goal_tolerance = 0.001;  % Original tight tolerance for v2
    max_trajectory_duration = 100;
    [Q_traj_JTDS, T_traj_JTDS] = computeFullTrajectory(q_init, x_target, motion_generator, ...
                                 goal_tolerance, max_trajectory_duration,options.orientation_flag);
end

% If someone wants to fix this for orientation then the
% 'computeFullTrajectory' function must be modified to consider the SO(3)
% space properly

% Use data-train here...
q_ref = Qs{selected_demo};
for dof = 1:7
    subplot(7,1,dof)  
    if using_v3
        % For v3, Q_traj_JTDS is [num_steps x dimq], so use column indexing
        plot(Q_traj_JTDS(:,dof),'-.','Color',[1 0 0], 'LineWidth',2);hold on;
    else
        % For v2, Q_traj_JTDS is [dimq x num_steps], so use row indexing
        plot(Q_traj_JTDS(dof,:),'-.','Color',[1 0 0], 'LineWidth',2);hold on;
    end
    plot(q_ref(dof,:),'-','Color', [0 0 0], 'LineWidth',2); hold on;
    xlabel('Time (samples)','Interpreter', 'LaTex', 'Fontsize', 15)
    ylabel('Angle (rad)','Interpreter', 'LaTex', 'Fontsize', 15)
    legend('learned','ref')
    grid on;
end
title(sprintf('Raw and Reconstructed Demonstrations for $q_%d$',dof), 'Interpreter', 'LaTex', 'Fontsize', 15)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Simulate task-space trajectory with JT-DS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract task-space positions and orientations
JTDS_task_space_traj_orient = zeros(6,length(Q_traj_JTDS));
JTDS_task_space_traj_pos    = zeros(3,length(Q_traj_JTDS));
for q=1:length(Q_traj_JTDS)
    if using_v3
        % For v3, Q_traj_JTDS is [num_steps x dimq], so use row indexing
        trans_tmp = robotplant.robot.fkine(Q_traj_JTDS(q,:));
    else
        % For v2, Q_traj_JTDS is [dimq x num_steps], so use column indexing and transpose
        trans_tmp = robotplant.robot.fkine(Q_traj_JTDS(:,q)');
    end
    JTDS_task_space_traj_pos(:,q) = trans_tmp(1:3,end);
    JTDS_task_space_traj_orient(:,q) = [trans_tmp(1:3,1); trans_tmp(1:3,2)];

end

% Plot Orientation trajectories
if options.orientation_flag    
    figure('Color',[1 1 1])
    scatter3(task_space_traj(1,:),task_space_traj(2,:),task_space_traj(3,:),20,'*'); hold on;
    scatter3(JTDS_task_space_traj_orient(1,:),JTDS_task_space_traj_orient(2,:),JTDS_task_space_traj_orient(3,:),20,'o','filled'); hold on;
    scatter3(task_space_traj(1,end),task_space_traj(2,end),task_space_traj(3,end),100,'o','filled','MarkerEdgeColor','k','MarkerFaceColor',[0 0 0]); hold on;
    scatter3(JTDS_task_space_traj_orient(1,end),JTDS_task_space_traj_orient(2,end),JTDS_task_space_traj_orient(3,end),100,'o','filled','MarkerEdgeColor','r','MarkerFaceColor',[1 0 0]); hold on;
    axis tight
    grid on;
    title('Task-Space trajectory (Orientation)', 'Interpreter', 'LaTex', 'Fontsize', 15)
    xlabel('$\theta_1$', 'Interpreter', 'LaTex', 'Fontsize', 15)
    ylabel('$\theta_2$', 'Interpreter', 'LaTex', 'Fontsize', 15)
    zlabel('$\theta_3$', 'Interpreter', 'LaTex', 'Fontsize', 15)
    err_orient = norm(JTDS_task_space_traj_orient(:,end) - task_space_traj(1:6,end))
end

% Plot Position trajectories
figure('Color',[1 1 1])
h1 = scatter3(task_space_traj(end-2,:),task_space_traj(end-1,:),task_space_traj(end,:),20,'*','DisplayName','Demonstration'); hold on;
h2 = scatter3(JTDS_task_space_traj_pos(1,:),JTDS_task_space_traj_pos(2,:),JTDS_task_space_traj_pos(3,:),20,'o','filled','DisplayName','JTDS Learned'); hold on;
h3 = scatter3(task_space_traj(end-2,end),task_space_traj(end-1,end),task_space_traj(end,end),100,'o','filled','MarkerEdgeColor','k','MarkerFaceColor',[0 0 0],'DisplayName','Demo End Point'); hold on;
h4 = scatter3(JTDS_task_space_traj_pos(1,end),JTDS_task_space_traj_pos(2,end),JTDS_task_space_traj_pos(3,end),100,'o','filled','MarkerEdgeColor','r','MarkerFaceColor',[1 0 0],'DisplayName','JTDS End Point'); hold on;
axis tight
grid on;
legend('show','Location','best');
title('Task-Space trajectory (Position)', 'Interpreter', 'LaTex', 'Fontsize', 15)
xlabel('$x_1$', 'Interpreter', 'LaTex', 'Fontsize', 15)
ylabel('$x_2$', 'Interpreter', 'LaTex', 'Fontsize', 15)
zlabel('$x_3$', 'Interpreter', 'LaTex', 'Fontsize', 15)
axis([-1 1 -1 1 -1 1])
err_pos = norm(JTDS_task_space_traj_pos(:,end) - task_space_traj(end-2:end,end))

% Special plot for v3 approach showing task space trajectory
if using_v3
    % --- DEBUGGING: Task Space DS Quality Check ---
    figure('Color',[1 1 1])
    
    % Extract task space demonstrations for comparison
    task_demo_pos = zeros(3, num);
    task_demo_vel = zeros(3, num);
    for i = 1:num
        q_demo = Data_train(1:dimq, i);
        qd_demo = Data_train(dimq+1:2*dimq, i);
        
        % Compute task space position and velocity from demo
        T_demo = robotplant.robot.fkine(q_demo');
        task_demo_pos(:, i) = T_demo(1:3, 4);
        
        J_demo = robotplant.robot.jacob0(q_demo);
        task_demo_vel(:, i) = J_demo(1:3, :) * qd_demo;
    end
    
    % Test task space DS at demonstration points
    task_ds_vel = zeros(3, num);
    for i = 1:num
        task_ds_vel(:, i) = motion_generator.eval_task_ds(task_demo_pos(:, i));
    end
    
    % Plot task space velocity comparison
    subplot(2,2,1)
    plot(task_demo_vel(1,:), 'b-', 'LineWidth', 2); hold on;
    plot(task_ds_vel(1,:), 'r--', 'LineWidth', 2);
    title('X-velocity: Demo vs DS')
    legend('Demo', 'DS')
    grid on;
    
    subplot(2,2,2)
    plot(task_demo_vel(2,:), 'b-', 'LineWidth', 2); hold on;
    plot(task_ds_vel(2,:), 'r--', 'LineWidth', 2);
    title('Y-velocity: Demo vs DS')
    legend('Demo', 'DS')
    grid on;
    
    subplot(2,2,3)
    plot(task_demo_vel(3,:), 'b-', 'LineWidth', 2); hold on;
    plot(task_ds_vel(3,:), 'r--', 'LineWidth', 2);
    title('Z-velocity: Demo vs DS')
    legend('Demo', 'DS')
    grid on;
    
    % Compute task space velocity RMSE
    task_vel_rmse = sqrt(mean(sum((task_demo_vel - task_ds_vel).^2, 1)));
    subplot(2,2,4)
    velocity_errors = sqrt(sum((task_demo_vel - task_ds_vel).^2, 1));
    plot(velocity_errors, 'k-', 'LineWidth', 2);
    title(sprintf('Task Velocity Error (RMSE: %.4f)', task_vel_rmse))
    ylabel('Error magnitude')
    grid on;
    
    fprintf('\n=== TASK SPACE DS DEBUGGING ===\n');
    fprintf('Task space velocity RMSE: %.6f m/s\n', task_vel_rmse);
    fprintf('Max velocity error: %.6f m/s\n', max(velocity_errors));
    fprintf('Mean velocity error: %.6f m/s\n', mean(velocity_errors));
    
    figure('Color',[1 1 1])
end

%% 
% --- DEBUGGING: Quick Task Space DS Quality Check ---
fprintf('\n=== QUICK TASK SPACE DS CHECK ===\n');

% Extract task space positions from demonstrations
n_check = min(100, size(Data_train, 2));  % Check first 100 points
X_demo_check = zeros(3, n_check);
Xd_demo_check = zeros(3, n_check);
Xd_ds_check = zeros(3, n_check);

for i = 1:n_check
    q = Data_train(1:dimq, i);
    qd = Data_train(dimq+1:2*dimq, i);
    
    % Get actual task space position and velocity
    T = robotplant.robot.fkine(q');
    X_demo_check(:, i) = T(1:3, 4);
    
    J = robotplant.robot.jacob0(q);
    Xd_demo_check(:, i) = J(1:3, :) * qd;
    
    % Get DS prediction
    Xd_ds_check(:, i) = motion_generator.eval_task_ds(X_demo_check(:, i));
end

% Compute quick metrics
task_vel_rmse_quick = sqrt(mean(sum((Xd_demo_check - Xd_ds_check).^2, 1)));
task_corr = corrcoef(Xd_demo_check(:), Xd_ds_check(:));

fprintf('Task space DS quality (first %d points):\n', n_check);
fprintf('  Velocity RMSE: %.6f m/s\n', task_vel_rmse_quick);
fprintf('  Velocity correlation R²: %.3f\n', task_corr(1,2)^2);
fprintf('  Attractor: [%.4f, %.4f, %.4f]\n', DS_params.att(1), DS_params.att(2), DS_params.att(3));
fprintf('  Demo end point: [%.4f, %.4f, %.4f]\n', X_demo_check(1,end), X_demo_check(2,end), X_demo_check(3,end));
fprintf('  Distance to attractor: %.6f m\n', norm(X_demo_check(:,end) - DS_params.att));

% Check velocity at attractor
vel_at_att = motion_generator.eval_task_ds(DS_params.att);
fprintf('  Velocity at attractor: %.6f m/s\n', norm(vel_at_att));

if task_vel_rmse_quick > 0.02
    fprintf('  ⚠️  High velocity RMSE - DS may not be learning well!\n');
end
if task_corr(1,2)^2 < 0.7
    fprintf('  ⚠️  Low velocity correlation - DS may not be learning well!\n');
end
if norm(vel_at_att) > 0.01
    fprintf('  ⚠️  High velocity at attractor - may not converge properly!\n');
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compare position and orientation from CPP simulation (JTDS) vs Training Data == THIS WORKS! %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Color',[1 1 1])
% JTDS_orientation = zeros(6,ceil(length(TheRobotTrajectory1)/10));
% for i=1:10:length(TheRobotTrajectory1)
%     JTDS_orientation(:,i) = TheRobotTrajectory1(i,1:6);
% end
% scatter3(task_space_traj(1,:),task_space_traj(2,:),task_space_traj(3,:),20,'*'); hold on;
% scatter3(JTDS_orientation(1,:),JTDS_orientation(2,:),JTDS_orientation(3,:),20,'o','filled'); hold on;
% scatter3(task_space_traj(1,end),task_space_traj(2,end),task_space_traj(3,end),100,'o','filled','MarkerEdgeColor','k','MarkerFaceColor',[0 0 0]); hold on;
% scatter3(JTDS_orientation(1,end),JTDS_orientation(2,end),JTDS_orientation(3,end),100,'o','filled','MarkerEdgeColor','r','MarkerFaceColor',[1 0 0]); hold on;
% axis tight
% grid on;
% title('Task-Space trajectory (Orientation)', 'Interpreter', 'LaTex', 'Fontsize', 15)
% xlabel('$\theta_1$', 'Interpreter', 'LaTex', 'Fontsize', 15)
% ylabel('$\theta_2$', 'Interpreter', 'LaTex', 'Fontsize', 15)
% zlabel('$\theta_3$', 'Interpreter', 'LaTex', 'Fontsize', 15)
% err_orient = norm(JTDS_orientation(:,end) - task_space_traj(1:6,end))
% 
% % Plot the simulated position and orientation
% figure('Color',[1 1 1])
% JTDS_position = zeros(3,ceil(length(TheRobotTrajectory1)/10));
% for i=1:10:length(TheRobotTrajectory1)
%     JTDS_position(:,i) = TheRobotTrajectory1(i,7:9)';
% end
% scatter3(task_space_traj(end-2,:),task_space_traj(end-1,:),task_space_traj(end,:),20,'*'); hold on;
% scatter3(JTDS_position(1,:),JTDS_position(2,:),JTDS_position(3,:),20,'o','filled'); hold on;
% scatter3(task_space_traj(end-2,end),task_space_traj(end-1,end),task_space_traj(end,end),100,'o','filled','MarkerEdgeColor','k','MarkerFaceColor',[0 0 0]); hold on;
% scatter3(JTDS_position(1,end),JTDS_position(2,end),JTDS_position(3,end),100,'o','filled','MarkerEdgeColor','r','MarkerFaceColor',[1 0 0]); hold on;
% axis tight
% grid on;
% title('Task-Space trajectory (Position)', 'Interpreter', 'LaTex', 'Fontsize', 15)
% xlabel('$\theta_1$', 'Interpreter', 'LaTex', 'Fontsize', 15)
% ylabel('$\theta_2$', 'Interpreter', 'LaTex', 'Fontsize', 15)
% zlabel('$\theta_3$', 'Interpreter', 'LaTex', 'Fontsize', 15)
% err_pos = norm(JTDS_position(:,end) - task_space_traj(end-2:end,end))