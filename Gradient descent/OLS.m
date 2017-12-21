clear;
x = csvread('MLR.csv',0,0,[0,0,999,29]);
y = csvread('MLR.csv',0,30);

beta_hat = (x.' * x)^(-1) * x.' * y;
true_beta = csvread('True_Beta.csv');
true_beta = true_beta.';
squared_error = (norm(beta_hat - true_beta))^2;

disp("The squared error is " + squared_error);