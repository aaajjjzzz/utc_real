function  pair = perm_comb(n)
% n个点里挑两个的排列组合
    pair = zeros(round(n*(n-1)/2),2);
    op = 0;
%     for num = 1:n
%             for i = 1: n
%                   H(i) = norm(X(num,:) - X(i,:));
%             end
%             [~, id] = sort(H);
            for k = 1:n
                for l = k+1:n
                  if k~=l
                      op = op+1;
                      pair(op,:) = [k,l];
                  end
                end
            end
    end
% end: I came, I saw, and I conquered.