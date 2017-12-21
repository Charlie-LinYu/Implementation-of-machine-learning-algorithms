clear;
%%read data;
x = csvread('MLR.csv',0,0,[0,0,999,29]);
y = csvread('MLR.csv',0,30);
beta_hat = (x.' * x)^(-1) * x.' * y;
true_beta = csvread('True_Beta.csv');
true_beta = true_beta.';

%%initialize some variables
l = max(eig(1/length(y) * x.' * x));
beta_est = zeros(30,1);
f_beta_hat = 1/(2*length(y))*(norm(y-x*beta_hat))^2;
diff_obj = zeros(1000,1);
diff_true_beta = zeros(1000,1);
diff_beta_hat = zeros(1000,1);

%%run the iterations for gradient descent algorithm and
%%calcuate the results required by queation (a) - (c)
for i = 1:1000
    beta_est = beta_est - 1/l * 1/length(y) * x.' * (x*beta_est-y);
    diff_obj(i) = log(1/(2*length(y))*(norm(y-x*beta_est))^2 - f_beta_hat);
    diff_true_beta(i) = (norm(beta_est - true_beta))^2;
    diff_beta_hat(i) = (norm(beta_est - beta_hat))^2;
end

index = 1:1000;
index = index.';

%%plot for the question (a), (b), (c) respectively
figure(1);
plot(index, diff_obj);
title('Figure for question (a)');
xlabel('Iteration Index (k)');
ylabel('$$log(f(\beta^{(k)})-f(\widehat{\beta}))$$','Interpreter','Latex');
pause;
figure(2);
plot(index, diff_true_beta);
title('Figure for question (b)');
xlabel('Iteration Index (k)');
ylabel('$$||\beta^{(k)} - \beta^\ast||^2_2$$','Interpreter','Latex');
pause;
figure(3);
plot(index, diff_beta_hat);
title('Figure for question (c)');
xlabel('Iteration Index (k)');
ylabel('$$||\beta^{(k)} - \widehat{\beta}||^2_2$$','Interpreter','Latex');
pause;
