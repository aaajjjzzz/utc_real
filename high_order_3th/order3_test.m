clc;clear;

%% 
%%%%%%%%%%%%%%%%%%Part 1 %%%%%%%%%%%%%%%%%%%%%%%%%%
% generate data
m = 90;
n = 1000;
knn = 5;

[data,gt] = generate_syndata(m,n);
it = 1; % resault 

%%
model_class = 3;
dim = 1000;

% 期望值
m = zeros(model_class,dim);  %m*d
vec = zeros(dim);

if model_class<=dim
    for i = 1:model_class
        m(i,i) = 2;
    end
else
    vec = zeros(1,dim);
    j=0;
    for i = 1:model_class
        if mod(j, dim) ~= 0 || j ==0
            vec_2 = vec;
            vec_2(j+1) = vec(j+1)+2;
            m(i,:) = vec_2;
            j = j+1;
        else
            vec = vec + 2;
            m(i,:) = vec;
            j=1;
        end
        
    end
end

% 协方差阵
for i = 1:model_class
    seed = randi(100)/100;
    s(:,:,i) = eye(dim)*seed;
    seed = randi([30,40]);
    num(i) = seed;
end

data_all = generate_data_GMM(dim, model_class, m, s, num);
% 
% 提出每组数据
% data1 = data((data(:,4) == 1), (1:3));
% data2 = data((data(:,4) == 2), (1:3));
data = data_all(:,1:dim);
gt = data_all(:,dim+1);
% 
% 
% load('F:\Work\Tensor Clustering\Tensor Spectral Clustering\Code-for-ETLMSC-master\data\data\yale.mat');
% data = X{1}';
% gt = double(gt);
% % gt = double(gt(1:55));
% % data = data(1:55,:);
cls_num = length(unique(gt));

%cluster = 3;
%cls_num = cluster;
cluster = cls_num;
num_cluster = cluster;
m = size(data,1);

%%
%% Kmeans
REPlic = 20; % Number of replications for KMeans
MAXiter = 1000; % Maximum number of iterations for KMeans 
groups = kmeans(data,num_cluster,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
Method_Name(it) = "Kmeans";
[ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups);
it = it+1;


% load('F:\Work\Tensor Clustering\Tensor Spectral Clustering\high_order\affinies_orders_v2_0815\mathwork\dataset\yale.mat');
% data = alldata';
% gt = true_gt';
% data= data(1:108,:);
% gt = gt(1:108);
% m = size(data,1);
% num_cluster = length(unique(gt));
% load('F:\Work\Tensor Clustering\Tensor Spectral Clustering\high_order\affinies_orders_v2_0815\mathwork\dataset\yale.mat');
% data = X{1}';
% gt = gt;
% %data= data(1:108,:);
% %gt = gt(1:108);
% m = size(data,1);
% num_cluster = length(unique(gt));
%%
%%%%%%%%%%%%%%%%%%Part 2 %%%%%%%%%%%%%%%%%%%%%%%%%
% 2-order similarity
L_2 = similarity_matrix(data);
L_hat_2 = eye(m)-L_2;
L_2_constructW = constructW(data);

S_cos = squareform(1-pdist(data,'cosine'));
D = sum(S_cos,1);
H = pdist2(data,data,'euclidean');
[L_4, embedding] = computing4tensor(data, knn, num_cluster);
L_3 = get_L3(data,knn,num_cluster);
[L3_m,L3_n] = size(L_3);

L_index = find(L_3);
for i = 1:L3_m
    for j = 1:L3_n
        if L_3(i,j) > 1e5
            L_3(i,j) = 0;
        end
    end
end


L_3_decom = Kr_product(L_2,L_2);


[u_1,s_1,vv_1] = svd(full(L_2));
V1_svd = vv_1(:,1:cls_num);
V2_svd = kron_col(V1_svd);

[ker_2,lamda_2] = eig(L_2);
[ker_2,lamda_2] = cdf2rdf(ker_2,lamda_2);
[max_lamda,lambda_index] = maxk(diag(lamda_2),cls_num);
V1_eig = ker_2(:,lambda_index);
% for i = 1:m
%     kerNS(i,:) = ker_N_2(i,:) ./ norm(ker_N_2(i,:)+eps);
% end
V2_eig = kron_col(V1_eig);

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
[~,groups_cons,CKSym] = SpectralClustering5(data,cls_num);
[ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_cons);
it = it+1;
%% undecom

opts.tol = 1e-4;
opts.max_iter = 100;
opts.rho = 1.1;
opts.mu = 1e0;
opts.max_mu = 1e2;
opts.DEBUG = 1;
opts.alpha = 1e-3;
Method_Name(it) = "ADMM_undecom_L24";
[V_1_ADMM_undecom,V_2_ADMM_undecom,iter_1,obj_rec_ADMM_undecom,difY1_rec1_undecom,difY2_rec1_undecom] = embedding_vector_ADMM_V2(L_2,L_4,cls_num,opts);
[~,group_admm_1,~] = SpectralClustering5(V_1_ADMM_undecom, cls_num);
[ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,group_admm_1);
it = it+1;




opts.tol = 1e-4;
opts.max_iter = 100;
opts.rho = 1.1;
opts.mu = 1e0;
opts.max_mu = 1e2;
opts.DEBUG = 1;
opts.alpha = 1e-3;
Method_Name(it) = "ADMM_undecom_L234";
[V_1_ADMM_undecom_v1v2v3,V_2_ADMM_undecom_v1v2v3,iter_2,obj_rec_ADMM_v1v2v3_undecom,difY1_rec1_undecom_v1v2v3,difY2_rec1_undecom_v1v2v3] = ADMM_V1V2V3(L_2,L_4,L_3,cls_num,opts);
[~,group_admm_1_v1v2v3,~] = SpectralClustering5(V_1_ADMM_undecom_v1v2v3, cls_num);
[ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,group_admm_1_v1v2v3);
it = it+1;

finale_resault = [Method_Name',ACC',ARI',F_SCORE',NMI',Purity'];
disp(finale_resault);
%%

save('syn_10cluster_0907.mat','data','gt','finale_resault','V_1_ADMM_undecom','V_2_ADMM_undecom','obj_rec_ADMM_undecom','difY1_rec1_undecom','difY2_rec1_undecom','V_1_ADMM_undecom_v1v2v3','V_2_ADMM_undecom_v1v2v3','iter_2','obj_rec_ADMM_v1v2v3_undecom','difY1_rec1_undecom_v1v2v3','difY2_rec1_undecom_v1v2v3')

%%
figure(1)
image(L_2,'CDataMapping','scaled');

figure(2)
L_2_V2 = V_1_ADMM_undecom*V_1_ADMM_undecom';
image(L_2_V2,'CDataMapping','scaled');

figure(3)
L_2_V2_max = V_1_ADMM_undecom(:,1)*V_1_ADMM_undecom(:,1)';
image(L_2_V2_max,'CDataMapping','scaled');

figure(4)
L_2_V2_max = V_1_ADMM_undecom(:,1)*V_1_ADMM_undecom(:,1)';
imagesc(L_2_V2_max)

figure(5)
L_2_V2_maxv123 = V_1_ADMM_undecom_v1v2v3(:,1)*V_1_ADMM_undecom_v1v2v3(:,1)';
imagesc(L_2_V2_maxv123)


