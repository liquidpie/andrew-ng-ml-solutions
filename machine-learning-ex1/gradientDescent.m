function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    old_theta = theta;
    theta_0 = 0;
    theta_1 = 0;
    for i = 1:m
         theta_0 += ((X(i, :) * old_theta) - y(i)) * X(i, 1); 
         theta_1 += ((X(i, :) * old_theta) - y(i)) * X(i, 2); 
    endfor
    theta(1) = theta(1) - (alpha * (1 / m) * theta_0); 
    theta(2) = theta(2) - (alpha * (1 / m) * theta_1);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf("Cost in iter %d is %f\n", iter, J_history(iter));
end

end
