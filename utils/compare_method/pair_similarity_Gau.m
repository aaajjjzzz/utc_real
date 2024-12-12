function S = pair_similarity_Gau(X)
% each row of X is a sample
% each colum of X is a feature
[ns,~]= size(X);
sigma = 10;
    for i = 1:ns
            for j = 1:ns
                for k = 1:ns
                    for l = 1:ns                          
                          S(k,l,i,j) = exp(-0.1*norm(X(i,:)-X(k,:),2)^2/(2*sigma^2))*exp(-0.1*norm(X(j,:)-X(l,:),2)^2/(2*sigma^2)); %Gaussian distance
                    end
                end
            end
     end
end