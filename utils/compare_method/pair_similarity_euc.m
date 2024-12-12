function S = pair_similarity_euc(X)
[ns,~]= size(X);
for i = 1:ns
     for j = 1:ns
         for k = 1:ns
             for l = 1:ns
               S(k,l,i,j) = exp(-0.1*norm(X(i,:)-X(k,:)))*exp(-0.1*norm(X(j,:)-X(l,:)));% Euclidean Diatance                          
             end
         end
     end
end
end
