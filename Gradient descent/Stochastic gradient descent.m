clear;
%%read data
x = csvread('MLR.csv',0,0,[0,0,999,29]);
y = csvread('MLR.csv',0,30);
beta_hat = (x.' * x)^(-1) * x.' * y;
true_beta = csvread('True_Beta.csv');
true_beta = true_beta.';

%%initialize variables
l = max(eig(1/length(y) * (x.') * x));
beta_est = zeros(30,1);
f_beta_hat = 1/(2*length(y))*(norm(y-x*beta_hat))^2;
index = 1:20;
index = index.';

%%call funtions to calculate required parameters and
%%plot them against passes regarding different step sizes
for k = [0.1,1.7,1,0.01]
    [diff_obj,diff_true_beta,diff_beta_hat] = get_diff(x,y,beta_est,true_beta,beta_hat,f_beta_hat,l,k);
    get_plot(index, diff_obj, diff_true_beta, diff_beta_hat, k);
    %%pause;
end
    
function [diff_obj,diff_true_beta,diff_beta_hat]=get_diff(x,y,beta_est,true_beta,beta_hat,f_beta_hat,l,step)
    diff_obj = zeros(20,1);
    diff_true_beta = zeros(20,1);
    diff_beta_hat = zeros(20,1);
    for i = 1:20                  %indicate passes
        for j = 1:1000            %indicate iterations
            beta_est = beta_est - step/l * x(j,:).' * (x(j,:) * beta_est - y(j));
        end
        diff_obj(i) = log(1/(2*length(y))*(norm(y-x*beta_est))^2 - f_beta_hat);
        diff_true_beta(i) = (norm(beta_est - true_beta))^2;
        diff_beta_hat(i) = (norm(beta_est - beta_hat))^2;
    end
end

function get_plot(index, diff_obj, diff_true_beta, diff_beta_hat, step)
    figure;
    %%subplot(1,3,1);
    plot(index, diff_obj);
    xlabel('Number of passes (k)');
    ylabel('$$log(f(\beta^{(k)})-f(\widehat{\beta}))$$','Interpreter','Latex');
    title(['Plot for (a) with step size of ' num2str(step) '/L']);
    pause;
    %%subplot(1,3,2);
    figure;
    plot(index, diff_true_beta);
    xlabel('Number of passes (k)');
    ylabel('$$||\beta^{(k)} - \beta^\ast||^2_2$$','Interpreter','Latex');
    title(['Plot for (b) with step size of ' num2str(step) '/L']);
    pause;
    %%subplot(1,3,3);
    figure;
    plot(index, diff_beta_hat);
    xlabel('Number of passes (k)');
    ylabel('$$||\beta^{(k)} - \widehat{\beta}||^2_2$$','Interpreter','Latex');
    title(['Plot for (c) with step size of ' num2str(step) '/L']);
    pause;
end