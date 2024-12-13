function finale_result = func_UTC(data,gt,dataset_name,save_path, opts_global)
%FUSED_DIFFOREDER fused different order similarity (pairwised tri ted)
%  

    lambda1 = 1;
    lambad2 = 1;
    order4_sim_opt = 1;
    L2_opt = 1;
    if isfield(opts_global,'lambda1');          lambda1 = opts_global.lambda1;   end
    if isfield(opts_global,'lambda2');          lambda2 = opts_global.lambda2; end
    if isfield(opts_global,'order4_sim_opt');  order4_sim_opt = opts_global.order4_sim_opt; end
    if isfield(opts_global,'L2_opt');          L2_opt = opts_global.L2_opt; end
    
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


    %%
    %%%%%%%%%%%%%%%%%%Part 2 %%%%%%%%%%%%%%%%%%%%%%%%%
    % 2-order similarity
    L_2 = similarity_matrix(data);
    L_2 = (L_2+L_2')/2;
    L_hat_2 = eye(m)-L_2;
    L_2_constructW = full(constructW(data));

    [L_4, embedding] = computing4(data,num_cluster,order4_sim_opt);%和论文不一样
    %L4 = (L_4+L_4')/2;
    L_4 = (L_4+L_4')/2;
    L_3 = get_L3(data,knn,num_cluster,1);
    [L3_m,L3_n] = size(L_3);
    
    for i = 1:L3_m
        for j = 1:L3_n
            if L_3(i,j) > 1e5
                L_3(i,j) = 0;
            end
        end
    end

    %%%%%%%%%%%Spectral cluster %%%%%%%%%%%%%
    % svd & eigen
    Method_Name(it) = "SC";
    [groups_svd,~] = Svd_Lap(L_hat_2,cls_num);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_svd);
    it = it+1;
    scatter(X(:,1),X(:,2),'filled','cdata',groups_svd);
    % ConstructW
    Method_Name(it) = "SC_ConstructW";
    [~,groups_cons,~] = SpectralClustering5(data,cls_num);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_cons);
    it = it+1;

    %%
    %% undecom
    %2+3 order
    cell_it= 1; embedding_bank={};results_bank = {};
    %lambda_2 = [0.05, 0.1, 0.25, 0.5, 0.75,1,1.5];
    if L2_opt==2
        L_2 = L_2_constructW;
    end
    
    opts.tol = 1e-3;
    opts.max_iter = 100;
    opts.rho = 1.1;
    opts.mu = 1e0;
    opts.max_mu = 1e2;
    opts.DEBUG = 0;
    opts.alpha = 1e-3;
    %2+3+4 order
    Method_Name(it) = "ADMM_undecom_L234_lambda1="+string(lambda1)+"_lambda2="+string(lambda2);
    [V_1_ADMM_L234,~,~,~,~,~] = ASL234_um(L_2,L_4,L_3,cls_num,opts,lambda1,lambda2);
    [~,group_L234_SC,~] = SpectralClustering5(V_1_ADMM_L234, cls_num);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,group_L234_SC);
    it = it+1;
    group_L234_Kmeans = kmeans(V_1_ADMM_L234,num_cluster,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    Method_Name(it) = "Kmeans_L234_lambda1="+string(lambda1)+"_lambda2="+string(lambda2);
    [ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,group_L234_Kmeans);
    it = it+1;
    embedding_bank{cell_it}  = [V_1_ADMM_L234];
    cell_it = cell_it + 1;
    
    
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

