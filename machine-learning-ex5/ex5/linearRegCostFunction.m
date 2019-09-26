function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
%m = length(y); % number of training examples
q = size(theta,1);
%X1 = [ones(m, 1) X];
thetamod = [0;theta(2:q)];

% You need to return the following variables correctly 
J = (1/(2*m))*sum((X * theta-y).^2)+lambda/(2*m)*(thetamod'*thetamod);
grad = (1/m)*X'*(X * theta-y)+(lambda/m)*thetamod;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
