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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


for i = 1:m
  sig = 1 / (1 + exp(-X(i,:) * theta))
  J += -y(i) * log(sig) - (1 - y(i)) * log(1-sig)
end

reg = 0
for j = 2:size(theta)
    reg += lambda * theta(j) * theta(j)
end

reg = reg / (2 * m)
J = (J / m) + reg

for j = 1:size(theta)
    for k = 1:m
        sig = 1 / (1 + exp(-X(k,:) * theta))
        grad(j) += (sig - y(k)) * X(k, j);
    end
    if (j != 1)
      grad(j) += lambda * theta(j)
    endif
end

grad = grad ./ m

% =============================================================

end
