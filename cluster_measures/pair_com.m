function  pair = pair_com(v,flag)
% ������v����������Ԫ���������
    [~,n] = size(v);
if flag == 1   %�зŻ���ѡ
     pair = zeros(n^2,2);
    op = 0;
    for k = 1:n
        for l = 1:n
                op = op + 1;
                pair(op,:) = [v(:,k),v(:,l)];
        end
    end
else           %�޷Ż���ѡ
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