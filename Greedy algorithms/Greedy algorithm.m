%In the train, valid and test data, remember the last column is label. Plot
%your figures in the defination of functions.
clear;
load('train-greedy.mat');
load('valid-greedy.mat');
load('test-greedy.mat');
load('true-beta.mat');



forward_greedy(train,validation,test,beta);
pause;
ridge_reg(train,validation,test,beta);
pause;
lasso_wrapper(train,validation,test,beta);
pause;
refined_est(train,validation,test,beta);

%Part a, implement the forward greedy algorithm
%Input: train data, validation data and test data
%Output: number of optimized features, optimal beta, estimation error and 
%        prediction error.
%        Plot your errors as iteration changes
function forward_greedy(train,validation,test,beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
times=100;                                 %the max numbre of k
A=[];
A_lag=[];
beta_est=zeros(size(train,2)-1,1);         
beta_lag=zeros(size(train,2)-1,1);
beta_res=zeros(size(train,2)-1,times);     %the result of beta with different K
x=train(:,1:size(train,2)-1);              %x from the training set
y=train(:,size(train,2));                  %y from the training set
x_v=validation(:,1:size(validation,2)-1);  %x from the validation set
y_v=validation(:,size(validation,2));      %y from the validation set
x_t=test(:,1:size(test,2)-1);        %x from the test set
y_t=test(:,size(test,2));            %y from the test set
val_error=zeros(times,1);                  %validation error  
est_error=zeros(times,1);                  %estimation error 
pred_error=zeros(times,1);                 %prediction error 

for k = 1:times
    tmp = x*beta_lag-y;
    obj_i = zeros(size(x,2),1);
    for j = 1:size(x,2)
        obj_i(j) = abs(x(:,j).' *tmp);
    end
    i = find(obj_i==max(obj_i),1);
    A=[A_lag;i];
    beta_noconstraints = (x(:,A).' * x(:,A))^(-1) *x(:,A).' * y;
    beta_est(A) = beta_noconstraints;
    beta_res(:,k) = beta_est;
    val_error(k) = norm(y_v-x_v*beta_est,2)^2;
    est_error(k) = norm(beta_est-beta,2);
    pred_error(k) = norm(y_t-x_t*beta_est,2)^2 /size(y_t,1);
    A_lag=A;
    beta_lag=beta_est;
    %disp(k);
end

optimal_k = find(val_error==min(val_error),1);
optimal_beta = beta_res(:,optimal_k);
optimized_features = find(optimal_beta~=0);
beta_final = [optimized_features optimal_beta(optimized_features)];
disp("For question (a)");
disp("Number of optimized features is " + size(optimized_features,1));
disp("The final optimal beta is:");
disp("Index    Optimal_beta")
disp(num2str(beta_final));
disp("Other elements of beta are 0");
disp("The optimized validation error is " + num2str(val_error(optimal_k)));
disp("The optimized estimation error is " + num2str(est_error(optimal_k)));
disp("The optimized prediction error is " + num2str(pred_error(optimal_k)));


k=(1:times).';
figure;
plot(k,val_error,'-o');
title("Validation errors along with K");
xlabel('K');
ylabel('Validation errors');
figure;
plot(k,est_error,'-o');
title("Estimation errors along with K");
xlabel('K');
ylabel('Estimation errors');
figure;
plot(k,pred_error,'-o');
title("Prediction errors along with K");
xlabel('K');
ylabel('Prediction errors');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


%Part b, implement the ridge regreesion estimator
%Input: train data, validation data and test data
%Output: optimal beta, optimal lambda, estimation error and prediction error.
%        Plot your errors as iteration changes
function ridge_reg(train,validation,test,beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%        Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda=[0.0125,0.025,0.05,0.1,0.2];
%lambda=[0.001:0.001:0.5];
%lambda=[0.125,0.25,0.5,1,2,4];
times = size(lambda,2);
beta_res=zeros(size(train,2)-1,times);     
x=train(:,1:size(train,2)-1);
y=train(:,size(train,2));
n=size(x,1);
x_v=validation(:,1:size(validation,2)-1);
y_v=validation(:,size(validation,2));
x_t=test(:,1:size(test,2)-1);
y_t=test(:,size(test,2));
val_error=zeros(1,times);                  
est_error=zeros(1,times);
pred_error=zeros(1,times);
for k = 1:times
    beta_est = (x.' * x + 2*n*lambda(k)*eye(size(x,2)))^(-1)*x.' * y;
    %beta_est = zeros(size(train,2)-1,1);
    %beta_est(998:1000,1) = (x(:,998:1000).' * x(:,998:1000) + lambda(k)*diag(ones(size(x(:,998:1000),2))))^(-1)*x(:,998:1000).' * y;
    beta_res(:,k) = beta_est;
    val_error(k) = norm(y_v-x_v*beta_est,2)^2;
    est_error(k) = norm(beta_est-beta,2);
    pred_error(k) = norm(y_t-x_t*beta_est,2)^2 /size(y_t,1);
    %disp(k);
end
optimal_k = find(val_error==min(val_error),1);
optimal_beta = beta_res(:,optimal_k);
optimized_features = find(optimal_beta~=0);
beta_final = [optimized_features optimal_beta(optimized_features)];
disp("For question (b)");
%disp("Number of optimized features is " + size(optimized_features,1));
disp("The final optimal beta is:");
disp("Index    Optimal_beta");
%disp(num2str(beta_final));
%disp(strjoin(arrayfun(@(x) num2str(x),optimal_beta.','UniformOutput',false),','));
%disp("Other elements of beta are 0");
optimal_lambda=lambda(optimal_k);
disp("Optimal lambda is " + optimal_lambda);
disp("The optimized validation error is " + num2str(val_error(optimal_k)));
disp("The optimized estimation error is " + num2str(est_error(optimal_k)));
disp("The optimized prediction error is " + num2str(pred_error(optimal_k)));

%k=(1:times).';
figure;
plot(lambda,val_error,'-o');
title("Validation errors along with \lambda");
xlabel('\lambda');
ylabel('Validation errors');
figure;
plot(lambda,est_error,'-o');
title("Estimation errors along with \lambda");
xlabel('\lambda');
ylabel('Estimation errors');
figure;
plot(lambda,pred_error,'-o');
title("Prediction errors along with \lambda");
xlabel('\lambda');
ylabel('Prediction errors');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%Part c, use lasso package to get optimized parameter.
%Input: train data, validation data and test data.
%Output: optimal beta, optimal lambda, estimation error and prediction
%        error. 
function lasso_wrapper(train,validation,test,beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=train(:,1:size(train,2)-1);
y=train(:,size(train,2));
x_v=validation(:,1:size(validation,2)-1);
y_v=validation(:,size(validation,2));
x_t=test(:,1:size(test,2)-1);
y_t=test(:,size(test,2));
[beta_res, fit_info] = lasso(x,y);
lambda = fit_info.Lambda;
intercept = fit_info.Intercept;
times = size(lambda,2);
val_error=zeros(times,1);                  %related to k
est_error=zeros(times,1);
pred_error=zeros(times,1);

for k = 1:times
    val_error(k) = norm(y_v-(x_v*beta_res(:,k)+intercept(k)),2)^2;
    est_error(k) = norm(beta_res(:,k)-beta,2);
    pred_error(k) = norm(y_t-(x_t*beta_res(:,k)+intercept(k)),2)^2 /size(y_t,1);
    %disp(k);
end
optimal_k = find(val_error==min(val_error),1);
optimal_beta = beta_res(:,optimal_k);
optimized_features = find(optimal_beta~=0);
beta_final = [optimized_features optimal_beta(optimized_features)];
%disp("Number of optimized features is " + size(optimized_features,1));
disp("For question (c)");
disp("The final optimal beta is:");
disp("Index    Optimal_beta")
disp(num2str(beta_final));
disp("Other elements of beta are 0");
optimal_lambda=lambda(optimal_k);
disp("Optimal lambda is " + optimal_lambda);
disp("The optimized validation error is " + num2str(val_error(optimal_k)));
disp("The optimized estimation error is " + num2str(est_error(optimal_k)));
disp("The optimized prediction error is " + num2str(pred_error(optimal_k)));

figure;
plot(lambda,val_error,'-o');
title("Validation errors along with \lambda");
xlabel('\lambda');
ylabel('Validation errors');
figure;
plot(lambda,est_error,'-o');
title("Estimation errors along with \lambda");
xlabel('\lambda');
ylabel('Estimation errors');
figure;
plot(lambda,pred_error,'-o');
title("Prediction errors along with \lambda");
xlabel('\lambda');
ylabel('Prediction errors');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end




%Part d, get your refined optimizer.
%Output: refined beta and estimation error.
function refined_est(train,validation,test,beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=train(:,1:size(train,2)-1);
y=train(:,size(train,2));
x_v=validation(:,1:size(validation,2)-1);
y_v=validation(:,size(validation,2));
x_t=test(:,1:size(test,2)-1);
y_t=test(:,size(test,2));
[beta_res, fit_info] = lasso(x,y);
lambda = fit_info.Lambda;
times = size(lambda,2);
val_error=zeros(times,1);                 

for k = 1:times
    val_error(k) = norm(y_v-x_v*beta_res(:,k),2)^2;
    %disp(k);
end
optimal_k = find(val_error==min(val_error),1);
beta_lasso = beta_res(:,optimal_k);
A = beta_lasso~=0;
beta_refit = zeros(size(beta_lasso,1),1);
beta_refit(A) = (x(:,A).' * x(:,A))^(-1) *x(:,A).' * y;

est_error_refit = norm(beta_refit-beta,2);
est_error_lasso = norm(beta_lasso-beta,2);

disp("For question (d)");
disp("The estimation error of Beta_refit is " + est_error_refit);
disp("The estimation error of Beta_lasso is " + est_error_lasso);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

