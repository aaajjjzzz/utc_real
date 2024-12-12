function [dk1, site1] = tensor_clustering_new3(X,H, knn)
% each row of X is a sample
% goal: find knn nearest neighbour of (num)th sample in X
    [n,~] = size(X);
%     n1 = round(factorial(knn)/(factorial(2)*factorial(knn-2)));
%     n2 = round(factorial(n1)/(factorial(2)*factorial(n1-2)));
    dk = cell(n,1);
    %dk = zeros(n*n2,1);
    site = cell(n,1);
    %site = zeros(n*n2,4);
    %op = 0;
%     sigma = 10;
    parfor num = 1:n
        dk{num} = [];
        site{num} = [];
        [~, id] = sort(H(num,:));% nlog_2(n)
        pair = pair_com(id(1:knn),2);% 1/2*k^2-k
        [p_row, ~] = size(pair);
        for i1 = 1:p_row-1 %1/8*k^4-1/2*k^3+1/4*k^2+1/2*k 
            for i2 = i1+1:p_row
                v1 = H(pair(i1,1),pair(i1,2));%  intra-cluster
                v2 = H(pair(i2,1),pair(i2,2));
                temp = (H(pair(i1,1),pair(i2,1))+H(pair(i1,2),pair(i2,2)))/(v1+v2+1e-4);% inter-cluster/intra-class
                dk{num} = [dk{num}; exp(-temp/mean(H(num,:)))];
                site{num} = [site{num};[pair(i1,1) pair(i2,1) pair(i1,2) pair(i2,2)]];
            end
        end
    end
dk1 = dk{1};
site1 = site{1};
for kk = 2:n
dk1 = [dk1;dk{kk}];
site1 = [site1; site{kk}];
end
end