function S = similarity_Gau(X)
% each row of X is a sample
% each colum of X is a feature
[ns,~]= size(X);
S = zeros(ns,ns);
Dis = pdist(X);
sigma = mean(Dis);
clear Dis
    for i = 1:ns
            for j = 1:ns                                    
                S(i,j) = exp(-norm(X(i,:)-X(j,:),2)^2/(2*sigma^2)); %Gaussian distance
            end
     end
end