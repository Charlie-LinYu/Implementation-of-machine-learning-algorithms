function [C, I, Loss] = myKmeans(X, K, maxIter, random_t)
%This is a function that performs K-means clustering
%The input is X: N*d, the input data
%             K: integer, number of clusters
%             maxIter: integer, number of iterations
%
%The output is C: K*d the center of clusters
%              I: N*1 the label of data
%              Loss: maxIter*1 within-cluster sum of squares (WCSS) in each
%              step

% number of vectors in X
[N, d] = size(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_all = zeros(N,random_t);
Loss_all = zeros(maxIter,random_t);

for t = 1:random_t
    % construct indicator matrix (each entry corresponds to the cluster
    % of each point in X)
    I = zeros(N, 1);

    % construct centers matrix
    %C = zeros(K, d);
    %random initialization
    C = min(X)+(max(X)-min(X)).*rand(K,d);

    % the list to record error
    Loss = zeros(maxIter,1);

    for i = 1:maxIter
        %step 1, assign the partition indicator to each point
        for j = 1:N
            error_j=zeros(1,K);
            for k = 1:K
                error_j(k)=norm(C(k,:)-X(j,:))^2;
            end
            I(j) = find(error_j==min(error_j),1,'first');
        end
        %step 2, reselect the centers of clusters by taking average
        for k=1:K
            C(k,:)=mean(X(I==k,:));
        end
        %calculate loss for each iteration
        error = 0;
        for j = 1:N
            error = error + norm(C(I(j),:)-X(j,:))^2;
        end
        Loss(i) = error;
    end
    I_all(:,t)=I;
    Loss_all(:,t)=Loss;
end

%find the result with the least loss
pos = find(Loss_all(maxIter,:)==min(Loss_all(maxIter,:)));
I = I_all(:,pos(1));
Loss = Loss_all(:,pos(1));
for k=1:K
    C(k,:)=mean(X(I==k,:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
