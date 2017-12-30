function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	%disp(theta);
	%disp(computeCost(X,y,theta));
	%temp=[alpha*[[[X*theta] -y]'*X] /m]';
	%disp('cal:');
	%disp(temp);
	%theta=theta-temp;
	disp('Cost:');
	disp(computeCost(X,y,theta));
	hyp=[X*theta];
	temp=zeros(2,1);
	for j=1:1:2
		
		for i=1:1:m
			temp(j)+= (hyp(i)-y(i))*X(i,j);
		end	
		theta(j)=theta(j)-alpha*1/m*temp(j);
		%disp(theta(j));
	end	

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
