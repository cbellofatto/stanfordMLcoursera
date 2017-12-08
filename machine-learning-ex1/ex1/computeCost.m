function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
err = zeros(m:1);

% ====================== OPTION 1 ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% for i = 1:m
% 	J = J + ((X(i,:) * theta) - y(i,:)) ^ 2;
% end

% J = J/(2*m);

% ====================== OPTION 2 =============================

err = X * theta;

J = sum((err .^ 2) + (y .^ 2) - (2 * err .* y))/(2*m);


% =========================================================================

end
