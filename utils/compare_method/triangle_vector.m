function [dk, ad]= triangle_vector(X,S)

% each row of X is a sample
   kk = 3;
   op2 = 0;
   [n,~] = size(X);
   ac = 34220;
   dk = zeros(ac,1);
   ad = zeros(size(dk,1),kk);
    for i1 = 1:n
        for j1 = i1+1:n
            for i2 = j1+1:n
                        op2 = op2 + 1;
                        dk(op2) = (S(i1,j1)+S(i1,i2)+S(j1,i2))/3;
                        ad(op2,:) = [i1,j1,i2];
            end
        end
    end
end