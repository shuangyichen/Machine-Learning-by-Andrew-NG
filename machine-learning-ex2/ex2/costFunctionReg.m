function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
q = size(theta,1);
thetamod = [0;theta(2:q)];
% You need to return the following variables correctly 
J = -(1/m)*sum(y.*log(sigmoid(X * theta))+(1-y).*log(1-sigmoid(X* theta)))+lambda/(2*m)*(thetamod'*thetamod);
grad = (1/m)*X'*(sigmoid(X * theta)-y)+lambda/m*thetamod;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
