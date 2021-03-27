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

% Part1 : Feedforward
X = [ones(m,1), X]; % 5000 X 401

a1 = X; % 5000 X 401
z2 = a1*Theta1' ; % 5000 X 401 times 401 X 25 -> 5000X25
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1), a2]; % 5000 X 26
z3 = a2*Theta2' ; % 5000 X 26 times 26 X 10 -> 5000 X 10 
H = sigmoid(z3); % 5000 X 10 

y_vec = zeros(m, num_labels); 
for i = 1:m
    y_vec(i,y(i)) = 1;
end

for i = 1: m
    J = J + (1/m)*sum((-y_vec(i,:).*log(H(i,:)))-((1-y_vec(i,:)).*log(1-H(i,:))));
end
J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

%Part2 : Back Propagation
Delta_1 = zeros(size(Theta1)); % 25 X 401
Delta_2 = zeros(size(Theta2)); % 10 X 26

d3 = H-y_vec; % 5000X10
d2 = (d3*Theta2)(:, 2:end).*sigmoidGradient(z2); % 5000 X 10 times 10 X 26 -> 5000 X 25

Delta_1 = d2'*a1; % 25 X 5000 times 5000 X 401 -> 25 X 401 
Delta_2 = d3'*a2; % 10 X 5000 times 5000 x 26  -> 10 x 26 

Theta1_grad = (1/m)*Delta_1;
Theta2_grad = (1/m)*Delta_2;
Theta1_grad(:, 2 : end) = Theta1_grad(:, 2 : end) .+ lambda * Theta1(:, 2 : end) / m;
Theta2_grad(:, 2 : end) = Theta2_grad(:, 2 : end) .+ lambda * Theta2(:, 2 : end) / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
