function Lap = similarity_matrix(data)
%SIMILARITY_MATRIX 此处显示有关此函数的摘要
%   Get data's lap matrix
%   D^{-0.5}*W*D^{-0.5}
    [m,n] = size(data);
    W = zeros(m,m);
    Dis = pdist(data);
    sigma = mean(Dis);
    for i = 1:m
        for j = 1:m
            W(i,j) = exp(-norm(data(i,:)-data(j,:))^2/(2*sigma^2));
        end
    end
    D = diag(sum(W,2));
    Lap = D^(-0.5) * W * D^(-0.5);
end

