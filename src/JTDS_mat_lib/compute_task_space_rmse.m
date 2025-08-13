function rmse_task = compute_task_space_rmse(motion_generator, Q_true, Qd_true, Xt_true, robotplant, orientation_flag)
% COMPUTE_TASK_SPACE_RMSE Compute RMSE in task space for trajectory prediction
% 
% Inputs:
%   motion_generator: The motion generator object
%   Q_true: True joint positions (dimq x N)
%   Qd_true: True joint velocities (dimq x N)  
%   Xt_true: True target task positions (dimx x N)
%   robotplant: Robot plant model
%   orientation_flag: Include orientation (1) or position only (0)
%
% Output:
%   rmse_task: RMSE in task space (m/s for position, rad/s for orientation)

N = size(Q_true, 2);
dimq = size(Q_true, 1);

if orientation_flag == 1
    task_dim = 6;  % Position + orientation
else
    task_dim = 3;  % Position only
end

Xd_pred = zeros(task_dim, N);
Xd_true = zeros(task_dim, N);

for i = 1:N
    q = Q_true(:, i);
    qd_true = Qd_true(:, i);
    x_target = Xt_true(:, i);
    
    % Get predicted joint velocity
    qd_pred = motion_generator.get_next_motion(q, x_target);
    
    % Compute Jacobian
    J = robotplant.robot.jacob0(q);
    if orientation_flag == 1
        J_task = J(1:6, :);
    else
        J_task = J(1:3, :);
    end
    
    % Compute task space velocities
    Xd_pred(:, i) = J_task * qd_pred;
    Xd_true(:, i) = J_task * qd_true;
end

% Compute RMSE in task space
error_task = Xd_pred - Xd_true;
rmse_task = sqrt(mean(sum(error_task.^2, 1)));

end
