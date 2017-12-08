function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
m = length(y);
n = length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

thetaReg = theta(2:n);

thetaReg = [0; thetaReg];

sig = sigmoid(X * theta);

J = (1/m) * ((-1 * y)' * log(sig)) - ((1 .- y)' * log(1 .- sig)) + ((lambda/(2 * m)) * (theta' * theta));

grad = (1/m) * X' * (sig - y) + (lambda/m) * thetaReg;

% =============================================================

end