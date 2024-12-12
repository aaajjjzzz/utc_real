function finale_result = func_ISP2(data,gt,dataset_name,save_path)
%FUNC_ISP2 此处显示有关此函数的摘要
%   此处显示详细说明
    X = data;
    m = size(data,1);
    cls_num = length(unique(gt));
    cluster = cls_num;
    num_cluster = cluster;

    it=1;
    knn=5;
    %%
    %% Kmeans
    REPlic = 20; % Number of replications for KMeans
    MAXiter = 1000; % Maximum number of iterations for KMeans
    groups = kmeans(data,num_cluster,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    Method_Name(it) = "Kmeans";
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups);
    it = it+1;


    %%
    %%%%%%%%%%%%%%%%%%Part 2 %%%%%%%%%%%%%%%%%%%%%%%%%
    % 2-order similarity
    L_2 = similarity_matrix(data);
    L_2 = (L_2+L_2')/2;
    L_hat_2 = eye(m)-L_2;
    L_2_constructW = full(constructW(data));
    % ConstructW
    Method_Name(it) = "ConstructW";
    [~,groups_cons,CKSym] = SpectralClustering5(data,cls_num);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_cons);
    it = it+1;
    
    S_cos = squareform(1-pdist(data,'cosine'));
    D = sum(S_cos,1);
    H = pdist2(data,data,'euclidean');
    [L_4, embedding] = computing4tensor(data, knn, num_cluster,1);
    L4 = (L_4+L_4')/2;

    [u_1,s_1,vv_1] = svd(full(L_2));
    V1_svd = vv_1(:,1:cls_num);
    V2_svd = kron_col(V1_svd);

    [ker_2,lamda_2] = eig(L_2);
    [ker_2,lamda_2] = cdf2rdf(ker_2,lamda_2);
    [max_lamda,lambda_index] = maxk(diag(lamda_2),cls_num);

    %%%%%%%%%%%Spectral cluster %%%%%%%%%%%%%
    % svd & eigen
    Method_Name(it) = "EigenVector";
    [groups_eig,~] = Eig_Lap(L_hat_2,cls_num);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_eig);
    it = it+1;
        %%%%%%%%%%4-order cluster %%%%%%%%%%%%
    Method_Name(it) = "PPS_1"; % embedding through computing4tensor ((dij+dil+dkj+dkl)/(dik+djl) + average)
    [~,group_pair3,~] = SpectralClustering5(embedding, cls_num);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,group_pair3);
    it = it+1;
    %%%%%%%%%4-order + 2-order %%%%%%%%%%%%
    Method_Name(it) = "IPS_1";
    [~,group_pair24,~] = SpectralClustering5(embedding+CKSym, cls_num);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,group_pair24);
    it = it+1;
    
        %%
    finale_result = [Method_Name',ACC',ARI',F_SCORE',NMI',Purity'];
    disp(finale_result);
    %save(save_path+file_stack(j).name+'_final_data.mat','data','gt','finale_result','V_1_ADMM_undecom','V_2_ADMM_undecom','obj_rec_ADMM_undecom','difY1_rec1_undecom','difY2_rec1_undecom','V_1_ADMM_undecom_v1v2v3','V_2_ADMM_undecom_v1v2v3','iter_2','obj_rec_ADMM_v1v2v3_undecom','difY1_rec1_undecom_v1v2v3','difY2_rec1_undecom_v1v2v3')
    xlswrite(save_path+string(dataset_name)+'_result.xls', finale_result);
end

