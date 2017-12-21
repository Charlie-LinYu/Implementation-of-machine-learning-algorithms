clear;
load Q1.mat

%% Implement part a
% with the variance matrix S, you can solve 
%the true first principle componenet v_real
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[a_u, a_s, a_v] = svd(S);
v_real = a_u(:,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('The ground truth PC is')
disp(v_real)

%% Implement part b
% with data X, you can solve 
%the empirical first principle componenet v_est
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[b_u, b_s, b_v] = svd(cov(X));
v_est = b_u(:,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('The empirical PC is')
disp(v_est)
%% Power Iteration
% you should implement this part in power_method.m
batchsize = length(X);
pass = 20;
[v1, lambda1,loss_real1, loss_est1] = power_method(X, batchsize, pass ,v_real, v_est);
myplot('Q1 c: Optimization error',loss_real1)
myplot('Q1 c: Estimation error',loss_est1)

%% Oja Iteration
% you should implement this part in oja_method.m
batchsize = 2;
pass = 1;
%eta = 0.1;
eta=0.0001;
[v2, lambda2,loss_real2, loss_est2] = oja_method(X, eta, batchsize, pass ,v_real, v_est);
myplot('Q1 d: Optimization error',loss_real2)
myplot('Q1 d: Estimation error',loss_est2)


function myplot(mytitle,y,x)
    figure
    if (nargin == 2)
        plot(y,'LineWidth',2,'MarkerSize',20);
    else
        plot(x,y,'LineWidth',2,'MarkerSize',20);
    end
    title(mytitle)
    set(gca,'FontSize',20)
end
