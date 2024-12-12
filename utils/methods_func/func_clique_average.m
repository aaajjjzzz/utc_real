function [weighted_L2, finale_result] = func_clique_average(data,gt, num_cluster, uniform_k)
%FUNC_CLIQUE_AVERAGE 此处显示有关此函数的摘要
%   此处显示详细说明
%%syn_data
    [m,~] = size(data);
    %%
    S = similarity_Gau(data);
    if uniform_k == 4
        [dk,ad] = pair2pair_vector(data,S);
    elseif uniform_k == 3
        [dk,ad]= triangle_vector(data,S);% 寻找每个点的K近邻，并在其中寻找两对pair，构建超边（i,j,k）
    end
    pair = perm_comb(m);% 寻找所有数据中点对（i,j）的位置信息
    % 构建超边与点对的match矩阵
    cmk = size(ad,1);
    cm2 = size(pair,1);
    Sp = zeros(cmk,cm2);  %超边与边的邻接矩阵 （n k）*(n 2)

    for i = 1:cm2   
        [row1 col1] = find(ad==pair(i,1));
        [row2 col2] = find(ad==pair(i,2));
        po = intersect(row1,row2);
        for j = 1:size(po)
            Sp(po(j),i) = 1;
        end
    end
    % 从中剥离出单点的信息
    d2 = lsqlin(Sp*uniform_k,dk,[],[],[],[],0,1);
    S3 = eye(m,m);
    for i = 1:cm2
        S3(pair(i,1),pair(i,2)) = d2(i);
        S3(pair(i,2),pair(i,1)) = d2(i);
    end
    %% Kmeans
    it = 1;

    Method_Name(it) = "CAVE_EigenVector";
    [groups_eig,~] = Eig_Lap_max(S3,num_cluster);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_eig);
    it = it+1;
    Method_Name(it) = "CAVE_SVD";
    [groups_svd,~] = Svd_Lap(S3,num_cluster);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_svd);
    it = it+1;
    finale_result = [Method_Name',ACC',ARI',F_SCORE',NMI',Purity'];
    disp(finale_result);
    weighted_L2 = S3;
end

