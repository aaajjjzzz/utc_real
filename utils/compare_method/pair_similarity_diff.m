function S = pair_similarity_diff(X,flag)
% each row of X is a sample
% each colum of X is a feature
[ns,~]= size(X);
sigma = 10;
    for i=1:ns
      for j=1:ns
          temp(1) = norm(X(i,:)-X(j,:));% Euclidean Diatance
          temp(2) = pdist([X(i,:);X(j,:)], 'chebychev');% Chebyshev Distance
          temp(3) = max(abs(X(i,:)-X(j,:)));% Manhattan Distance
          temp(4) = norm(X(i,:)-X(j,:),2)^2/(2*sigma^2); %Gaussian distance
        S(i,j) = temp(flag);
      end
    end
   S = exp(-0.1 * S); 
end

