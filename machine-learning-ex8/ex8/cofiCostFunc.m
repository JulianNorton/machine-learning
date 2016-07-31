function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%



for i = 1:num_movies
    for j = 1:num_users
        if (R(i, j) == 1)
           J += (Theta(j, :) * X(i, :)' - Y(i, j))^2
        endif
    end
end

J = J / 2

J += (lambda * (sum(sum(Theta(:, 1:end) .* Theta(:, 1:end))) + \
                sum(sum(X(:, 1:end) .* X(:, 1:end))))) / 2

for i = 1:num_movies
    for j = 1:num_features
        for k = 1:num_users
            if (R(i, k) == 1)
              X_grad(i, j) += (Theta(k, :) * X(i, :)' - Y(i, k)) * \
                              Theta(k, j)
            endif
        end
        X_grad(i, j) += lambda * X(i, j)
    end
end


for i = 1:num_users
    for j = 1:num_features
        for k = 1:num_movies
            if (R(k, i) == 1)
              Theta_grad(i, j) += (Theta(i, :) * X(k, :)' - Y(k, i)) \
                                  * X(k, j)
            endif
        end
        Theta_grad(i, j) += lambda * Theta(i, j)
    end
end













% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
