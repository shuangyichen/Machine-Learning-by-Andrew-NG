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
X = [ones(m, 1) X];
X1 = sigmoid(X*Theta1');
X1 = [ones(m,1),X1];
X2 = sigmoid(X1*Theta2');
temp = zeros(m,num_labels);
for j = 1:size(y,1)
temp(j,y(j)) = 1;
end
y = temp;
Theta11 = Theta1';
Theta22 = Theta2';
[p,q] = size(Theta11);
[k,l] = size(Theta22);
theta1mod = [zeros(1,q);Theta11(2:p,:)];
theta2mod = [zeros(1,l);Theta22(2:k,:)];


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%for i = 1:1:num_labels
%J = (-1/m)*sum(sum(y.*log(X2)+(1-y).*log(1-X2)))+(lambda/(2*m))*(sum(sum(theta1mod'*theta1mod))+sum(sum(theta2mod'*theta2mod)));
J = (-1/m)*sum(sum(y.*log(X2)+(1-y).*log(1-X2)))+(lambda/(2*m))*(sum(sum(theta1mod.^2))+sum(sum(theta2mod.^2)));
 
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
Del1 = 0;
Del2 = 0;
for t = 1:1:m
    a1 = X(t,:);
    z2 = a1*Theta1';
    a2 = sigmoid(z2);
    a2 = [1 a2];
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    delta3 = a3 - y(t,:);
    temp32 = delta3*Theta2;
    delta2 = temp32(2:end).*sigmoidGradient(z2);
    Del1 = Del1 + delta2'*a1;
    Del2 = Del2 + delta3'*a2;
end
for i = 1:1:size(Theta1_grad,2)
    if(i==1)
        Theta1_grad(:,i) = (1/m)*Del1(:,i);
    else
        Theta1_grad(:,i) = (1/m)*Del1(:,i)+(lambda/m)*Theta1(:,i);
    end
end
for j = 1:1:size(Theta2_grad,2)
    if(j==1)
        Theta2_grad(:,j) = (1/m)*Del2(:,j);
    else
        Theta2_grad(:,j) = (1/m)*Del2(:,j)+(lambda/m)*Theta2(:,j);
    end
end


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
