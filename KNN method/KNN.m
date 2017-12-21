%In this problem, you need to implement KNN algorithm.
%The inputs: data from train-KNN and test-KNN
%            In the datasets, the first column is your horizontal coordinate(x axis), 
%            the second column is your vertical coordinate(y axis), 
%            the third column represents the data label.       
%
%
%The outputs: plot of original data,
%             plots of your classification results after implementing KNN
%             algorithm with K= 1, 2, 5, 20,
%             error rate of classification results with K= 1, 2, 5, 20.
%              
%Load the training data and plot it.
%To better recognize your results during grading process, here you are
%required to use "o" to represent label 1, use "+" to represent label 2, 
%and use "*" to represent label 3.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
train=load('train-KNN.mat');
train=train.a;
figure(1);
gscatter(train(:,1),train(:,2),train(:,3),'rgb','o+*');
title('Plot for training set');
xlabel('X');
ylabel('Y');
pause;
k=transpose([1,2,5,20]);
error_l1=zeros(size(k,1),1);
error_l2=zeros(size(k,1),1);

%calculate the training error with respect to L1 and L2 norm
for i = 1:size(k,1)
    res_l1=implement_knn_l1(train, train, k(i));
    res_l2=implement_knn_l2(train, train, k(i));
    error_l1(i)=sum(res_l1~=train(:,3))/size(train,1);
    error_l2(i)=sum(res_l2~=train(:,3))/size(train,1);
end
figure(2);
plot(k, error_l1);
title('Training error with L1 norm');
xlabel('K');
ylabel('Traing error');
pause;

figure(3);
plot(k, error_l2);
title('Training error with L2 norm');
xlabel('K');
ylabel('Traing error');
pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%Implement KNN algorithm with your testing data(remember you need to consider
%both l1 distance and l2 distance) and save the classification results in
%test_label_l1 and test_label_l2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test=load('test-KNN.mat');
test=test.B;
%k=transpose([1,2,5,20]);
test_label_l1 = zeros(size(test,1),size(k,1));
test_label_l2 = zeros(size(test,1),size(k,1));

for i = 1:size(k,1)
    test_label_l1(:,i)=implement_knn_l1(train, test, k(i));
    test_label_l2(:,i)=implement_knn_l2(train, test, k(i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Plot your classification results (i.e. test_label_l1 & test_label_l2), 
%See Lecture 2 page 43/47 as a reference.
%Remember to indicate your distance type and K values
%To better recognize your results during grading process, here you are
%required to use "o" to represent label 1, use "+" to represent label 2, 
%and use "*" to represent label 3.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(k,1)
    figure;
    %subplot(2,1,1);
    gscatter(test(:,1),test(:,2),test_label_l1(:,i),'rgb','o+*');
    title(['Plot for testing set with L1 norm and K = ' num2str(k(i))]);
    xlabel('X');
    ylabel('Y');
    pause;
    %subplot(2,1,2);
    figure;
    gscatter(test(:,1),test(:,2),test_label_l2(:,i),'rgb','o+*');
    title(['Plot for testing set with L2 norm and K = ' num2str(k(i))]);
    xlabel('X');
    ylabel('Y');
    pause;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [knn_result] = implement_knn_l1(train_data, test_data, k)
    knn_result=zeros(size(test_data,1),1);
    distance = zeros(size(train_data,1),1);
    distance = [distance train_data(:,3)];   %associate distance with labels
    for i = 1:size(test_data)
        distance(:,1) = sum(abs(train_data(:,1:2)-test_data(i,1:2)),2);
        distance_sorted = sortrows(distance, 1);
        c1 = sum(distance_sorted(1:k,2)==1);
        c2 = sum(distance_sorted(1:k,2)==2);
        c3 = sum(distance_sorted(1:k,2)==3);
        class = [c1 c2 c3];
        knn_result(i) = find(class==max(class),1);
    end     
end

function [knn_result] = implement_knn_l2(train_data, test_data, k)
    knn_result=zeros(size(test_data,1),1);
    distance = zeros(size(train_data,1),1);
    distance = [distance train_data(:,3)];
    for i = 1:size(test_data)
        distance(:,1) = sqrt(sum((train_data(:,1:2)-test_data(i,1:2)).^2,2));
        distance_sorted = sortrows(distance, 1);
        c1 = sum(distance_sorted(1:k,2)==1);
        c2 = sum(distance_sorted(1:k,2)==2);
        c3 = sum(distance_sorted(1:k,2)==3);
        class = [c1 c2 c3];
        knn_result(i) = find(class==max(class),1);
    end     
end