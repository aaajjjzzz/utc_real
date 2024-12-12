function S = pair_similarity_che(X)
% each row of X is a sample
% each colum of X is a feature
[ns,~]= size(X);
sigma = 10;
    for i = 1:ns
            for j = 1:ns
                for k = 1:ns
                    for l = 1:ns                          
                          S(k,l,i,j) = exp(-0.1*pdist([X(i,:);X(k,:)],'chebychev'))*exp(-0.1*pdist([X(j,:);X(l,:)],'chebychev'));% Chebyshev Distance
                    end
                end
            end
     end
end