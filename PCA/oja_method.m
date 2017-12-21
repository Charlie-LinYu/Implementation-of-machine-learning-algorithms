function [ v, lambda, loss_real, loss_est] = oja_method ( X, eta, batchsize, pass ,real_w, est_w)
%This is a function that performs stochastic oja iteration
%The input is X: N*d, all the input data
%             eta: float, the step size
%             batchsize: integer, number of data for a covariance matrix
%             pass: integer, number of passes through data
%             real_w: d, the ground truth PC, to help you record the loss
%             est_w: d, the empirical PC, to help you record the loss
%         
%The output is v: d, the PC
%              lambda: float, largest eigenvalue
%              loss_real, loss_est: maxIter*1, record the loss path,
%              in order to generate the plot. Note you need to calculate
%              maxIter yourself

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%parameters initialization
maxIter = 0.4*1/eta;
v_prior = ones(size(X,2),1) * 1/sqrt(size(X,2));
v_pass = zeros(size(X,2),pass);
v = zeros(size(X,2),1);
loss_real = zeros(maxIter,1);
loss_est = zeros(maxIter,1);
for i = 1: maxIter
    for k = 1:pass
        sample=randperm(length(X),batchsize);
        %disp(sample);
        %A = cov(X(randperm(length(X),batchsize),:));
        A = 0.5*(X(sample(1),:)-X(sample(2),:)).' *(X(sample(1),:)-X(sample(2),:));
        v_pass(:,k) = (v_prior+eta*A*v_prior)/norm(v_prior+eta*A*v_prior);
    end
    v = mean(v_pass,2);
    if v(1)*real_w(1)<0
        v=-v;
    end
    loss_real(i) = log(norm(real_w-v)^2);
    loss_est(i) = log(norm(est_w-v)^2);
    v_prior = v;
end

lambda = v.' * cov(X) * v;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end