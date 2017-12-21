function [ v, lambda, loss_real, loss_est] = power_method ( X, batchsize, pass,real_w, est_w)

%This is a function that performs stochastic power iteration
%The input is X: N*d, all the input data
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
v_prior = ones(size(X,2),1) * 1/sqrt(size(X,2));
v = zeros(size(X,2),1);
loss_real = zeros(pass,1);
loss_est = zeros(pass,1);
for i = 1: pass
    A = cov(X(randperm(length(X),batchsize),:));
    v = A*v_prior/norm(A*v_prior);
    if v(1)*real_w(1)<0
        v=-v;
    end
    loss_real(i) = log(norm(real_w-v)^2);
    %loss_est(i) = log(norm(est_w-v)^2);
    loss_est(i) = (norm(est_w-v)^2);
    v_prior = v;
end

lambda = v.' * cov(X) * v;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end