function H = pair_similarity(h2,S)
[ns,~]= size(h2);
H = zeros(ns,ns,ns,ns);
% decomposable
for i=1:ns
    for j=1:ns
        for k=1:ns
            for l=1:ns
               H(i,k,j,l) = S(i,j)*S(k,l);
            end
         end
    end
end
