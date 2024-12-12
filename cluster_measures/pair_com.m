function  pair = pair_com(v,flag)
% 列向量v中任意两个元素排列组合
    [~,n] = size(v);
if flag == 1   %有放回挑选
     pair = zeros(n^2,2);
    op = 0;
    for k = 1:n
        for l = 1:n
                op = op + 1;
                pair(op,:) = [v(:,k),v(:,l)];
        end
    end
else           %无放回挑选
  pair = zeros(round(factorial(n)/(factorial(2)*factorial(n-2))),2);
  op = 0;
    for k = 1:n
        for l = k+1:n
                op = op + 1;
                pair(op,:) = [v(:,k),v(:,l)];
        end
    end
end
end