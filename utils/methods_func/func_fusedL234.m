function finale_result = func_fusedL234(data,gt,dataset_name,save_path, order4_sim_opt)
%FUSED_DIFFOREDER fused different order similarity (pairwised tri ted)
%  

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

    S_cos = squareform(1-pdist(data,'cosine'));
    D = sum(S_cos,1);
    H = pdist2(data,data,'euclidean');
    [L_4, embedding] = computing4tensor(data, knn, num_cluster,order4_sim_opt);
    L4 = (L_4+L_4')/2;
    L_3 = get_L3(data,knn,num_cluster,4);
    L_3_gc = get_L3(data,knn,num_cluster,3);
    [L3_m,L3_n] = size(L_3);

    L_index = find(L_3);
    for i = 1:L3_m
        for j = 1:L3_n
            if L_3(i,j) > 1e5
                L_3(i,j) = 1;
            end
        end
    end
    %%%%%%%%%%%Spectral cluster %%%%%%%%%%%%%
    % svd & eigen
    Method_Name(it) = "EigenVector";
    [groups_eig,~] = Eig_Lap(L_hat_2,cls_num);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_eig);
    it = it+1;
    Method_Name(it) = "SVD";
    [groups_svd,~] = Svd_Lap(L_hat_2,cls_num);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_svd);
    it = it+1;
    % ConstructW
    Method_Name(it) = "ConstructW";
    [~,groups_cons,~] = SpectralClustering5(data,cls_num);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_cons);
    it = it+1;

    %%
    %% undecom
    %2+3 order
    lambda = [1:1:16]; cell_it= 1; embedding_bank={};results_bank = {};

    %for loop_1 = 1:2
    for loop_2 = 1:size(lambda,2)
        %lambda1 = lambda(loop_1);
        lambda2 = lambda(loop_2);
        opts.tol = 1e-4;
        opts.max_iter = 100;
        opts.rho = 1.1;
        opts.mu = 1e0;
        opts.max_mu = 1e2;
        opts.DEBUG = 1;
        opts.alpha = 1e-3;
        %2+3+4 order
            Method_Name(it) = "ADMM_undecom_L234_lambda1="+"_lambda2="+string(lambda2);
            [V_1_ADMM_L234,V_2_ADMM_L234,iter_L234,obj_rec_ADMM_L234,difY1_rec1_L234,difY2_rec1_L234] = ADMM_V1V3(L_2,L_3,cls_num,opts,lambda2);
            [~,group_L234_SC,~] = SpectralClustering5(V_1_ADMM_L234, cls_num);
            [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,group_L234_SC);
            it = it+1;
        embedding_bank{cell_it}  = [V_1_ADMM_L234];
        results_bank{cell_it} = [group_L234_SC];
        cell_it = cell_it + 1;
        
    end
    %end
    %%
    finale_result = [Method_Name',ACC',ARI',F_SCORE',NMI',Purity'];
    disp(finale_result);
    %save(save_path+file_stack(j).name+'_final_data.mat','data','gt','finale_result','V_1_ADMM_undecom','V_2_ADMM_undecom','obj_rec_ADMM_undecom','difY1_rec1_undecom','difY2_rec1_undecom','V_1_ADMM_undecom_v1v2v3','V_2_ADMM_undecom_v1v2v3','iter_2','obj_rec_ADMM_v1v2v3_undecom','difY1_rec1_undecom_v1v2v3','difY2_rec1_undecom_v1v2v3')
    xlswrite(save_path+string(dataset_name)+'_result.xls', finale_result);
    save(save_path+string(dataset_name)+'_result.mat','data','gt','finale_result','embedding_bank', 'results_bank','dataset_name');
    
    %% 
    %visual
    %k = func_visual(data, gt, embedding_bank, dataset_name, save_path);
    
end

