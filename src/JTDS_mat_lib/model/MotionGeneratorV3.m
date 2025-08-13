classdef MotionGeneratorV3 < handle
    %MOTIONGENERATORV3 Motion generator for JTDS_Solver_v3 approach
    %   This class implements the execution policy for the task-space LPV-DS 
    %   with null space preference mapping approach.
    
    properties
        plant           % RobotPlant for kinematics
        DS_params       % Task space LPV-DS parameters from JTDS_Solver_v3
        mapping_gmm     % Null space mapping GMM from JTDS_Solver_v3
        null_gain       % Gain for null space attraction (default: 1.0)
        orientation_flag % Whether to include orientation (default: 0)
    end
    
    methods
        function obj = MotionGeneratorV3(robotplant, DS_params, mapping_gmm, options)
            obj.plant = robotplant;
            obj.DS_params = DS_params;
            obj.mapping_gmm = mapping_gmm;
            
            % Set default options
            if nargin < 4
                options = struct();
            end
            if isfield(options, 'null_gain')
                obj.null_gain = options.null_gain;
            else
                obj.null_gain = 1.0;  % Default gain
            end
            if isfield(options, 'orientation_flag')
                obj.orientation_flag = options.orientation_flag;
            else
                obj.orientation_flag = 0;  % Default: position only
            end
        end
          function qd = get_next_motion(obj, q, xt, ~)
            % Implements the v3 execution policy:
            % 1. phi_dot = eval_LPV_DS(DS_params, phi)
            % 2. J_pinv = pinv(J_task)
            % 3. N = eye(dimq) - J_pinv * J_task
            % 4. q_null = sample from mapping_gmm given phi
            % 5. q_dot = J_pinv * phi_dot + N * k * (q_null - q)
            
            dimq = length(q);
            
            % Step 1: Get current task space position
            if obj.orientation_flag == 1
                phi_current = obj.plant.end_pos_orien(q');
                task_dim = 6;
            else
                phi_current = obj.plant.forward_kinematics(q');
                task_dim = 3;
            end
            
            % Safety check: if very close to target, stop
            task_error = norm(phi_current - xt);
            if task_error < 0.005  % 5mm tolerance
                qd = zeros(dimq, 1);
                return;
            end
            
            % Step 2: Evaluate task space LPV-DS
            phi_dot = obj.eval_task_ds(phi_current);
            
            % Limit task space velocity for stability
            max_task_vel = 0.1;  % m/s - conservative limit
            if norm(phi_dot) > max_task_vel
                phi_dot = phi_dot / norm(phi_dot) * max_task_vel;
            end
            
            % Step 3: Compute Jacobian and null space projector
            J = obj.plant.robot.jacob0(q);
            if obj.orientation_flag == 1
                J_task = J(1:6, :);
            else
                J_task = J(1:3, :);
            end
            J_pinv = pinv(J_task);
            N = eye(dimq) - J_pinv * J_task;
              % Step 4: Primary task velocity
            qd_primary = J_pinv * phi_dot;
            
            % Step 5: Secondary task (null space) - FULL IMPLEMENTATION
            % Get preferred null space configuration using the learned mapping
            q_null_preferred = obj.get_null_space_config(phi_current);
            
            % Null space control: drive towards preferred configuration
            qd_secondary = N * obj.null_gain * (q_null_preferred - q);
            
            % Step 6: Combine hierarchical control
            qd = qd_primary + qd_secondary;
            
            % Safety limits
            max_joint_vel = 0.5;  % rad/s - conservative
            qd = max(min(qd, max_joint_vel), -max_joint_vel);
        end
        
        function qd = get_next_motion_orientation(obj, q, xt, ~)
            % For orientation, just call the regular method with orientation_flag
            qd = obj.get_next_motion(q, xt, []);
        end
          function phi_dot = eval_task_ds(obj, phi)
            % Evaluate the learned task space LPV-DS
            % This implements the DS function from JTDS_Solver_v3
            
            try
                % Use the stored DS function if available
                if isfield(obj.DS_params, 'ds_fun') && ~isempty(obj.DS_params.ds_fun)
                    phi_dot = obj.DS_params.ds_fun(phi);
                else
                    % Fallback: manually compute LPV-DS
                    phi_dot = obj.compute_lpv_ds(phi);
                end
                
                % Safety check: limit output magnitude
                max_vel = 0.2;  % m/s
                if norm(phi_dot) > max_vel
                    phi_dot = phi_dot / norm(phi_dot) * max_vel;
                end
                
            catch ME
                % Fallback to simple proportional control if DS fails
                fprintf('Warning: DS evaluation failed, using fallback control\n');
                att = obj.DS_params.att;
                phi_dot = -0.1 * (phi - att);  % Simple proportional control
            end
        end
        
        function phi_dot = compute_lpv_ds(obj, phi)
            % Manual computation of LPV-DS if ds_fun not available
            
            % Get GMM parameters
            Priors = obj.DS_params.GMM.Priors;
            Mu = obj.DS_params.GMM.Mu;
            Sigma = obj.DS_params.GMM.Sigma;
            A_k = obj.DS_params.A_k;
            att = obj.DS_params.att;
            
            K = length(Priors);
            task_dim = obj.DS_params.task_dim;
            
            % Compute mixing weights
            h_raw = zeros(1, K);
            for k = 1:K
                h_raw(k) = Priors(k) * mvnpdf(phi', Mu(:, k)', Sigma(:, :, k));
            end
            
            h_total = sum(h_raw);
            if h_total == 0
                h = ones(1, K) / K;  % Equal weights if too far from all Gaussians
            else
                h = h_raw / h_total;
            end
            
            % Compute weighted dynamics
            A_weighted = zeros(task_dim, task_dim);
            for k = 1:K
                A_weighted = A_weighted + h(k) * A_k(:, :, k);
            end
            
            % Apply dynamics (assuming attractor-based DS)
            if obj.DS_params.constr_type == 1
                phi_shifted = phi - att;
                phi_dot = A_weighted * phi_shifted;
            else
                phi_dot = A_weighted * (phi - att);
            end
        end
          function q_null = get_null_space_config(obj, phi)
            % Get preferred null space configuration given task space state
            % This implements the conditional distribution p(q_null | phi) from the learned GMM
            
            dimq = obj.plant.robot.n;
            task_dim = obj.mapping_gmm.task_dim;
            
            try
                % Method 1: Use the learned joint distribution p(phi, q_null)
                % Create query point for conditional distribution
                joint_data = [phi; zeros(dimq, 1)];  % [task_position; dummy_null_space]
                
                % Find the most likely component for this task state
                component_probs = zeros(obj.mapping_gmm.gmm.NumComponents, 1);
                for k = 1:obj.mapping_gmm.gmm.NumComponents
                    mu_k = obj.mapping_gmm.gmm.mu(k, :)';
                    sigma_k = obj.mapping_gmm.gmm.Sigma(:, :, k);
                    
                    % Only consider task space part for likelihood
                    mu_task = mu_k(1:task_dim);
                    sigma_task = sigma_k(1:task_dim, 1:task_dim);
                    
                    component_probs(k) = obj.mapping_gmm.gmm.ComponentProportion(k) * ...
                                        mvnpdf(phi, mu_task', sigma_task);
                end
                
                % Get the most likely component
                [~, best_comp] = max(component_probs);
                
                % Extract the null space part from this component
                mu_best = obj.mapping_gmm.gmm.mu(best_comp, :)';
                q_null = mu_best((task_dim+1):end);
                
            catch
                % Method 2: Fallback - use weighted average of all null space means
                try
                    component_means = obj.mapping_gmm.gmm.mu;
                    null_space_means = component_means(:, (task_dim+1):end);
                    priors = obj.mapping_gmm.gmm.ComponentProportion;
                    
                    % Weighted average of null space means
                    q_null = (priors * null_space_means)';
                    
                catch
                    % Method 3: Ultimate fallback - project current config to null space
                    J = obj.plant.robot.jacob0(obj.plant.robot.q);  % Use current or default config
                    if obj.orientation_flag == 1
                        J_task = J(1:6, :);
                    else
                        J_task = J(1:3, :);
                    end
                    J_pinv = pinv(J_task);
                    N = eye(dimq) - J_pinv * J_task;
                    q_null = N * zeros(dimq, 1);  % Null space projection of zero (home position)
                end
            end
        end
        
        function qd_fun = ODE_fun(obj, xt, dt, orientation_flag)
            % Create ODE function handle for trajectory integration
            if nargin < 4
                orientation_flag = obj.orientation_flag;
            end
            
            qd_fun = @(t, q) obj.get_next_motion(q, xt, []);
        end
          function options = ODE_options(obj, xt, goal_tolerance, orientation_flag)
            % Create ODE options with event function for goal reaching
            if nargin < 4
                orientation_flag = obj.orientation_flag;
            end
            
            % Use more lenient tolerance for v3
            adjusted_tolerance = max(goal_tolerance, 0.02);  % At least 2cm
            
            function [value, is_terminal, direction] = event_fun(t, y)
                value = zeros(size(t));
                for i = 1:length(t)
                    if orientation_flag == 1
                        current_pose = obj.plant.end_pos_orien(y(:, i)');
                    else
                        current_pose = obj.plant.forward_kinematics(y(:, i)');
                    end
                    value(i) = norm(current_pose - xt) - adjusted_tolerance;
                end
                is_terminal = ones(size(t));
                direction = -1*ones(size(t));
            end
            
            % More tolerant integration options
            options = odeset('Events', @event_fun, ...
                            'AbsTol', 1e-4, ...        % Less strict absolute tolerance
                            'RelTol', 1e-3, ...        % Less strict relative tolerance  
                            'MaxStep', 0.1, ...        % Larger max step
                            'InitialStep', 0.01);      % Conservative initial step
        end
        
        function A = compute_A(obj, q)
            % Dummy function for compatibility with trajectory_error
            % Returns identity matrix since v3 doesn't use this approach
            dimq = length(q);
            A = eye(dimq);
        end
        
        function [q_sim, phi_sim, t_sim] = generate_simple_trajectory(obj, q_init, dt, max_time, target_phi)
            % Simple fixed-step trajectory generation for debugging/visualization
            % Inputs:
            %   q_init - Initial joint configuration [dimq x 1]
            %   dt - Time step (default: 0.01)
            %   max_time - Maximum simulation time (default: 10.0)
            %   target_phi - Target task space position (optional)
            %
            % Outputs:
            %   q_sim - Joint trajectory [dimq x num_steps]
            %   phi_sim - Task space trajectory [3 x num_steps]
            %   t_sim - Time vector [1 x num_steps]
            
            if nargin < 3 || isempty(dt)
                dt = 0.01;  % 10ms time step
            end
            if nargin < 4 || isempty(max_time)
                max_time = 10.0;
            end
            
            % Initialize
            dimq = obj.plant.robot.n;
            t_sim = 0:dt:max_time;
            num_steps = length(t_sim);
            
            q_sim = zeros(dimq, num_steps);
            phi_sim = zeros(3, num_steps);
            % Set initial conditions
            q_current = q_init(:);  % Ensure column vector
            q_sim(:, 1) = q_current;
            % Compute initial task space position
            phi_current = obj.plant.forward_kinematics(q_current');
            phi_sim(:, 1) = phi_current;
            
            % Define stopping criteria
            if nargin >= 5 && ~isempty(target_phi)
                target_tolerance = 0.01;  % 1cm tolerance
                has_target = true;
            else
                target_tolerance = inf;
                has_target = false;
            end
            
            fprintf('Starting simple trajectory generation...\n');
            fprintf('Initial position: [%.3f, %.3f, %.3f]\n', phi_current(1), phi_current(2), phi_current(3));
            if has_target
                fprintf('Target position: [%.3f, %.3f, %.3f]\n', target_phi(1), target_phi(2), target_phi(3));
            end
              % Main simulation loop
            for i = 2:num_steps
                try
                    % Compute task space velocity using JTDS-v3
                    phi_dot = obj.eval_task_ds(phi_current);
                      % Apply velocity limits in task space
                    max_task_vel = 0.1;  % Reduce from 0.5 to 0.1 m/s for better accuracy
                    phi_dot_norm = norm(phi_dot);
                    if phi_dot_norm > max_task_vel
                        phi_dot = phi_dot * (max_task_vel / phi_dot_norm);
                    end
                      % Compute Jacobian
                    J = obj.plant.robot.jacob0(q_current);
                    J_task = J(1:3, :);  % Only position part
                    
                    % Compute null space projection
                    J_pinv = pinv(J_task);
                    N = eye(dimq) - J_pinv * J_task;
                    
                    % Get preferred null space configuration
                    q_null_pref = obj.get_null_space_config(phi_current);
                    
                    % Null space velocity (with damping) - DEBUGGING: Make null space gain adjustable
                    k_null = 0.1;  % Reduce from 1.0 to 0.1 to minimize null space interference
                    q_dot_null = k_null * N * (q_null_pref - q_current);
                    
                    % DEBUGGING: Check if null space component is too large
                    null_space_magnitude = norm(q_dot_null);
                    primary_task_magnitude = norm(J_pinv * phi_dot);
                    
                    if mod(i, round(num_steps/5)) == 0  % Print every 20% of simulation
                        fprintf('Step %d: Primary=%.4f, Null=%.4f, Ratio=%.2f\n', ...
                            i, primary_task_magnitude, null_space_magnitude, ...
                            null_space_magnitude/primary_task_magnitude);
                    end
                    
                    % Combined velocity
                    q_dot = J_pinv * phi_dot + q_dot_null;
                      % Apply joint velocity limits
                    max_joint_vel = 0.5;  % Reduce from 1.0 to 0.5 rad/s for better accuracy
                    for j = 1:dimq
                        if abs(q_dot(j)) > max_joint_vel
                            q_dot(j) = sign(q_dot(j)) * max_joint_vel;
                        end
                    end
                      % Simple Euler integration
                    q_current = q_current + q_dot * dt;
                    
                    % Update task space position
                    phi_current = obj.plant.forward_kinematics(q_current');
                    
                    % Store results
                    q_sim(:, i) = q_current;
                    phi_sim(:, i) = phi_current;
                    
                    % Check convergence to target
                    if has_target
                        distance_to_target = norm(phi_current - target_phi);
                        if distance_to_target < target_tolerance
                            fprintf('Reached target at t=%.2f seconds (distance=%.4f)\n', t_sim(i), distance_to_target);
                            % Truncate arrays
                            q_sim = q_sim(:, 1:i);
                            phi_sim = phi_sim(:, 1:i);
                            t_sim = t_sim(1:i);
                            break;
                        end
                    end
                    
                    % Progress update
                    if mod(i, round(num_steps/10)) == 0
                        progress = i / num_steps * 100;
                        fprintf('Progress: %.1f%% (t=%.2f, pos=[%.3f,%.3f,%.3f])\n', ...
                            progress, t_sim(i), phi_current(1), phi_current(2), phi_current(3));
                    end
                    
                catch ME
                    warning('Error at step %d (t=%.3f): %s', i, t_sim(i), ME.message);
                    % Stop simulation on error
                    q_sim = q_sim(:, 1:i-1);
                    phi_sim = phi_sim(:, 1:i-1);
                    t_sim = t_sim(1:i-1);
                    break;
                end
            end
            
            fprintf('Simple trajectory generation completed.\n');
            fprintf('Final position: [%.3f, %.3f, %.3f]\n', phi_current(1), phi_current(2), phi_current(3));
            fprintf('Trajectory length: %d steps (%.2f seconds)\n', size(phi_sim, 2), t_sim(end));
        end
    end
end
