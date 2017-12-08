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




% Initialize vectors of activation layers

a_1 = zeros(input_layer_size + 1);
a_2 = zeros(hidden_layer_size);
a_3 = zeros(num_labels);
z_2 = zeros(length(a_2));
z_3 = zeros(length(a_3));

d_2 = zeros(length(a_2));
d_3 = zeros(length(a_3));

Delta_2 = zeros(length(a_3), length(a_2)+1);
Delta_1 = zeros(length(a_2), length(a_1));

% Add bias terms to X and transpose for vectorization

X = [ones(m,1) X];
X = X';

% Forward prop through each training example i

for i = 1:m

   % Forward propagate through each example to generate z and activation layers

   a_1 = X(:,i);
   z_2 = Theta1 * a_1;
   a_2 = [1 ; sigmoid(z_2)];
   z_3 = Theta2 * a_2;
   a_3 = sigmoid(z_3);

   % Transform each correct answer y(i) into a vector.

   yvec = zeros(num_labels,1);
   yvec(y(i)) = 1;

   % Calculate deltas

   d_3 = a_3 - yvec;
   d_2 = (Theta2'(2:end,:) * d_3) .* sigmoidGradient(z_2);

   % Accumulate gradients

   Delta_2 = Delta_2 + (d_3 * a_2');
   Delta_1 = Delta_1 + (d_2 * a_1');

% Calculate cost function
   J = J + ( ((-1 * yvec)' * log(a_3)) - ((1 .- yvec)' * log(1 .- a_3)) );

endfor

Theta2_grad = (1 / m) * Delta_2;
Theta1_grad = (1 / m) * Delta_1;
J = J/m;

% Add regularization

Theta1s = Theta1(:, 2:end);
Theta2s = Theta2(:, 2:end);

J = J + ( lambda/ (2 * m) ) * ( sum(Theta1s(:) .* Theta1s(:)) + sum(Theta2s(:) .* Theta2s(:)) );

   Theta2_grad = Theta2_grad + (lambda / m) * [zeros(size(Theta2s(:,1))) Theta2s];
   Theta1_grad = Theta1_grad + (lambda / m) * [zeros(size(Theta1s(:,1))) Theta1s];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
