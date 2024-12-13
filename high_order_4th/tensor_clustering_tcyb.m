function [dk1, site1] = tensor_clustering_tcyb(X,H, knn,opt_sim)
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
    %parfor num = 1:n
    if opt_sim == 1 % pair wised
        for num = 1:n
            dk{num} = [];
            site{num} = [];
            [~, id] = sort(H(num,:));% nlog_2(n)   第num行距离排序 距离v_{num}最近的点
            pair = pair_com(id(1:knn),2);% 1/2*k^2-k   挑选出（k-1）！个点对，离v_{num}最近的k-1个点顺序选择
            [p_row, ~] = size(pair);
            for i1 = 1:p_row-1 %1/8*k^4-1/2*k^3+1/4*k^2+1/2*k   % (i1,1)->i;(i1,2)->k;(i2,1)->j;(i2,2)->l;
                for i2 = i1+1:p_row
                    d_ik = H(pair(i1,1),pair(i1,2));%  intra-cluster: i k
                    d_jl = H(pair(i2,1),pair(i2,2)); % j l
                    den = d_ik + d_jl + 1e-4;
                    d_ij = H(pair(i1,1),pair(i2,1));
                    d_kl = H(pair(i1,2),pair(i2,2));
                    numerator = d_ij + d_kl;
                    temp = numerator/den;% inter-cluster/intra-class
                    dk{num} = [dk{num}; exp(-temp/mean(H(num,:)))];
                    site{num} = [site{num};[pair(i1,1) pair(i2,1) pair(i1,2) pair(i2,2)]];
                end
            end
        end
    elseif opt_sim==2  % picture singular value
        for num = 1:n
            dk{num} = [];
            site{num} = [];
            [~, id] = sort(H(num,:));% nlog_2(n)   第num行距离排序 距离v_{num}最近的点
            tri_id = nchoosek(id(1:knn/2),3);
            tri_id_num = size(tri_id);
            for i = 1:tri_id_num
                x1 = X(num,:);
                x2 = X(tri_id(i,1),:);
                x3 = X(tri_id(i,2),:);
                x4 = X(tri_id(i,3),:);
                pic_M = [x1' x2' x3' x4'];
                [U S V] = svd(pic_M);
                sim = S(1,1)^2/norm(sum(S,1),'F')^2;
                dis = S(4,4)^2/norm(sum(S,1),'F')^2;
                dk{num} = [dk{num}; sim];
                site{num} = [site{num};[num tri_id(i,1) tri_id(i,2) tri_id(i,3)]];
            end
        end
    end
dk1 = dk{1};
site1 = site{1};
for kk = 2:n
    dk1 = [dk1;dk{kk}];   %四个点之间相似度
    site1 = [site1; site{kk}]; %对应的索引
end
end