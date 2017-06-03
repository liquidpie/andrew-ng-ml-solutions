function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% recode the labels as vectors containing only values 0 or 1
label_vector = [1:num_labels];
yv = (y == label_vector);


% ----- Hypothesis -----
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
h_x = sigmoid(z3);

% ----- Cost Function -----

for k = 1:num_labels
  yk = yv(:, k);
  h_xk = h_x(:, k);
  J += sum((-yk .* log(h_xk)) - ((1 - yk) .* log(1 - h_xk)));
endfor

J = J ./ m;

% add regularization 
reg_cost = 0;

% no. of input and output units for theta1
in_unit = size(Theta1, 2);
out_unit = size(Theta1, 1);

for j = 1:out_unit
  for k = 2:in_unit
    reg_cost += Theta1(j,k)**2;
  endfor
endfor

% no. of input and output units for theta2
in_unit = size(Theta2, 2);
out_unit = size(Theta2, 1);

for j = 1:out_unit
  for k = 2:in_unit
    reg_cost += Theta2(j,k)**2;
  endfor
endfor

reg_cost = lambda * (1 / (2 * m)) * reg_cost;

J += reg_cost;


% ----- Gradient Descent using backpropagation -----
for i = 1:m
  % Error term for layer 3
  delta_3 = h_x(i, :)' - yv(i, :)'; % 10 X 1
  
  % Error term for layer 2
  delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2(i, :)']);
  delta_2 = delta_2(2:end, :);
  
  % Accumulate gradient for layer 2
  Theta2_grad += delta_3 * a2(i, :); % 10 X 26

  % Accumulate gradient for layer 1
  Theta1_grad += delta_2 * a1(i, :); % 25 X 401
  
endfor

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

% add regularization
Theta1_grad(:, 2:end) += lambda * (1 / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += lambda * (1 / m) * Theta2(:, 2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
