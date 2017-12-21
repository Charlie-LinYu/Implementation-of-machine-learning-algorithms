clear;
%%parameters setting
load('spamdata.mat');
x=training_set;
y=training_set_label;
x_test = testing_set;
y_test = testing_set_label;
%%
%%for GDA method
p = sum(y)/length(y);
average_0 = zeros(1,48);
average_1 = zeros(1,48);
covariance=zeros(48,48);

%%calculate the miu vectors for y=0 and y=1
for i = 1:length(y)
    average_0 = average_0 + x(i,:)*(1-y(i));
    average_1 = average_1 + x(i,:)*y(i);
end
average_0 = average_0/(length(y) - sum(y));
average_1 = average_1/sum(y);

%%calculate the covariance matrix
for i = 1:length(y)
    if y(i) == 1
        covariance = covariance + (x(i,:)-average_1).' *(x(i,:)-average_1);
    else
        covariance = covariance + (x(i,:)-average_0).' *(x(i,:)-average_0);
    end
end
covariance = covariance/length(y);

%%use the function mvnpdf to get the joint pdf of 48 variables per training observation
%%and imply the Bayes formula
y_hat = (mvnpdf(x,average_1,covariance)*p) ./ ...
    (mvnpdf(x,average_1,covariance)*p + mvnpdf(x,average_0,covariance)*(1-p));
y_hat(y_hat>=0.5) = 1;
y_hat(y_hat~=1) = 0;
training_error = sum(abs(y_hat-y))/length(y);
disp("The training error of GDA is " + training_error);

%%use the function mvnpdf to get the joint pdf of 48 variables per testing observation
%%and imply the Bayes formula
y_predict = (mvnpdf(x_test,average_1,covariance)*p) ./ ...
    (mvnpdf(x_test,average_1,covariance)*p + mvnpdf(x_test,average_0,covariance)*(1-p));
y_predict(y_predict>=0.5) = 1;
y_predict(y_predict~=1) = 0;
testing_error = sum(abs(y_predict-y_test))/length(y_test);
disp("The testing error of GDA is " + testing_error);
%%
clearvars -except x y x_test y_test;
%%for NB-GDA method
p = sum(y)/length(y);
average_0 = zeros(1,48);
average_1 = zeros(1,48);
variance=zeros(1,48);

%%calculate the miu vectors for y=0 and y=1
for i = 1:length(y)
    average_0 = average_0 + x(i,:)*(1-y(i));
    average_1 = average_1 + x(i,:)*y(i);
end
average_0 = average_0/(length(y) - sum(y));
average_1 = average_1/sum(y);

%%calculate the variance vector for each variable and
%%convert it to a diagonal covariance matrix
for i = 1:length(y)
    if y(i) == 1
        variance = variance + (x(i,:)-average_1) .* (x(i,:)-average_1);
    else
        variance = variance + (x(i,:)-average_0) .* (x(i,:)-average_0);
    end
end
variance = variance/length(y);
variance = diag(variance);

%%the product of independent conditional probability of 
%%48 variables per training observation is equal to 
%%the joint pdf of them with a diagonal covariance matrix,
%%and then imply the Bayes formula
y_hat = (mvnpdf(x,average_1,variance)*p)./ ...
    (mvnpdf(x,average_1,variance)*p + mvnpdf(x,average_0,variance)*(1-p));
y_hat(y_hat>=0.5) = 1;
y_hat(y_hat~=1) = 0;
training_error = sum(abs(y_hat-y))/length(y);
disp("The training error of NB-GDA is " + training_error);

%%the product of independent conditional probability of 
%%48 variables per testing observation is equal to 
%%the joint pdf of them with a diagonal covariance matrix,
%%and then imply the Bayes formula
y_predict = (mvnpdf(x_test,average_1,variance)*p) ./ ...
    (mvnpdf(x_test,average_1,variance)*p + mvnpdf(x_test,average_0,variance)*(1-p));
y_predict(y_predict>=0.5) = 1;
y_predict(y_predict~=1) = 0;
testing_error = sum(abs(y_predict-y_test))/length(y_test);
disp("The testing error of NB-GDA is " + testing_error);
%%
clearvars -except x y x_test y_test;
%%for NB-BDA method
%%make some modification of the value of variables
x_BDA = x/100;
x_BDA(x_BDA~=0) = 1;
x_test_BDA = x_test/100;
x_test_BDA(x_test_BDA~=0) = 1;
p = sum(y)/length(y);
gamma_0 = zeros(1,48);
gamma_1 = zeros(1,48);
y_hat = zeros(length(y),1);
y_predict = zeros(size(x_test_BDA,1),1);

%%calculate the probability vectors for the Bernoulli distribution regarding
%%y=0 and y=1
for i = 1:length(y)
    gamma_0 = gamma_0 + x_BDA(i,:)*(1-y(i));
    gamma_1 = gamma_1 + x_BDA(i,:)*y(i);
end
gamma_0 = gamma_0/(length(y) - sum(y));
gamma_1 = gamma_1/sum(y);
gamma_0(gamma_0 == 0) = 0.01;
gamma_1(gamma_1 == 0) = 0.01;

%%calculate the product of independent conditional probability of 
%%48 variables per training observation, and imply the Bayes formula
for i = 1:length(y)
    p1 = prod(x_BDA(i,:) .* gamma_1 + (ones(1,48)-x_BDA(i,:)) .* (ones(1,48)-gamma_1));
    p0 = prod(x_BDA(i,:) .* gamma_0 + (ones(1,48)-x_BDA(i,:)) .* (ones(1,48)-gamma_0));
    if p1*p/(p1*p + p0*(1-p)) >= 0.5
        y_hat(i)=1;
    else
        y_hat(i)=0;
    end
end
training_error = sum(abs(y_hat-y))/length(y);
disp("The training error of NB-BDA is " + training_error);

%%calculate the product of independent conditional probability of 
%%48 variables per testing observation, and imply the Bayes formula
for i = 1:length(y_test)
    p1 = prod(x_test_BDA(i,:) .* gamma_1 + (ones(1,48)-x_test_BDA(i,:)) .* (ones(1,48)-gamma_1));
    p0 = prod(x_test_BDA(i,:) .* gamma_0 + (ones(1,48)-x_test_BDA(i,:)) .* (ones(1,48)-gamma_0));
    if p1*p/(p1*p + p0*(1-p)) >= 0.5
        y_predict(i)=1;
    else
        y_predict(i)=0;
    end
end
testing_error = sum(abs(y_predict-y_test))/length(y_test);
disp("The testing error of NB-BDA is " + testing_error);
%%
clearvars -except x y x_test y_test;
%%for QDA method
p = sum(y)/length(y);
average_0 = zeros(1,48);
average_1 = zeros(1,48);
covariance_0=zeros(48,48);
covariance_1=zeros(48,48);

%%calculate the miu vectors for y=0 and y=1
for i = 1:length(y)
    average_0 = average_0 + x(i,:)*(1-y(i));
    average_1 = average_1 + x(i,:)*y(i);
end
average_0 = average_0/(length(y) - sum(y));
average_1 = average_1/sum(y);

%%calculate two covariance matrixes regarding y=0 and y=1
for i = 1:length(y)
    if y(i) == 1
        covariance_1 = covariance_1 + (x(i,:)-average_1).' *(x(i,:)-average_1);
    else
        covariance_0 = covariance_0 + (x(i,:)-average_0).' *(x(i,:)-average_0);
    end
end
covariance_0 = covariance_0/(length(y) - sum(y));
covariance_1 = covariance_1/sum(y);

%%to make sure the covariance matrixes is not singular
for i = 1:48
    covariance_0(i,i) = covariance_0(i,i) + 0.01;
    covariance_1(i,i) = covariance_1(i,i) + 0.01;
end

%%use the function mvnpdf to get the joint pdf of 48 variables per training observation
%%and imply the Bayes formula
y_hat = (mvnpdf(x,average_1,covariance_1)*p)./ ...
    (mvnpdf(x,average_1,covariance_1)*p + mvnpdf(x,average_0,covariance_0)*(1-p));
y_hat(y_hat>=0.5) = 1;
y_hat(y_hat~=1) = 0;
training_error = sum(abs(y_hat-y))/length(y);
disp("The training error of QDA is " + training_error);

%%use the function mvnpdf to get the joint pdf of 48 variables per testing observation
%%and imply the Bayes formula
y_predict = (mvnpdf(x_test,average_1,covariance_1)*p) ./ ...
    (mvnpdf(x_test,average_1,covariance_1)*p + mvnpdf(x_test,average_0,covariance_0)*(1-p));
y_predict(y_predict>=0.5) = 1;
y_predict(y_predict~=1) = 0;
testing_error = sum(abs(y_predict-y_test))/length(y_test);
disp("The testing error of QDA is " + testing_error);