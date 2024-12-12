function finale_resault_fused = func_decomposeing(data,gt,file_name,save_path)
%FUNC_DECOMPOSEING 此处显示有关此函数的摘要
%   此处显示详细说明
X = data;
m = size(data,1);
cls_num = length(unique(gt));
cluster = cls_num;
num_cluster = cls_num;
it =1;
cell_it=1;embedding_bank  = {};results_bank = {};
knn = 5;
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
L_hat_2 = eye(m)-L_2;
L_2_constructW = full(constructW(data));


H = pdist2(data,data,'euclidean');
[L_4, embedding] = computing4tensor(data, knn, num_cluster);
L4 = (L_4+L_4')/2;
L_3 = get_L3(data,knn,num_cluster,1);
L_3_gc = get_L3(data,knn,num_cluster,3);
[L3_m,L3_n] = size(L_3);
for i = 1:L3_m
    for j = 1:L3_n
        if L_3(i,j) > 1e5
            L_3(i,j) = 0;
        end
    end
end
[L3_m,L3_n] = size(L_3_gc);
for i = 1:L3_m
    for j = 1:L3_n
        if L_3_gc(i,j) > 1e5
            L_3_gc(i,j) = 0;
        end
    end
end


%%%%%%%%%%%Spectral cluster %%%%%%%%%%%%%
% svd & eigen
Method_Name(it) = "EigenVector";
[groups_eig,embedding_eigen] = Eig_Lap(L_hat_2,cls_num);
[ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_eig);
it = it+1;
embedding_bank{cell_it}  = [embedding_eigen];
results_bank{cell_it} = [groups_eig];
cell_it = cell_it + 1;
Method_Name(it) = "SVD";
[groups_svd,~] = Svd_Lap(L_hat_2,cls_num);
[ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_svd);
it = it+1;
% ConstructW
Method_Name(it) = "ConstructW";
[~,groups_cons,CKSym] = SpectralClustering5(data,cls_num);
[ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_cons);
it = it+1;

%%
% decomposing the L4 and L3
[L4_dec,L4_dec_2,~] = NearestKroneckerProduct(full(L_4), [m,m],[m,m]);  %L_4 = kron(L4_dec,L4_dec_2) 
[L3_dec,L3_dec_2] = Kr_approximating(full(L_3));                        %L_3 = kr_product(L3_dec,L3_dec_2)
[L3_gc_dec,L3_gc_dec_2] = Kr_approximating(full(L_3_gc));                             
lambda = [0.5,1];


max_flag =0;
L4_V1 = eigen_vec(L4_dec,cls_num, max_flag);    %v41
L4_V2 = eigen_vec(L4_dec_2,cls_num,max_flag);   %v42 
L3_V1 = eigen_vec(L3_dec,cls_num,max_flag);     %v31
L3_V2 = eigen_vec(L3_dec_2,cls_num,max_flag);   %v32

%% 
C31 = L3_V1(:,1)'*L3_dec*L3_V1(:,1);    %v31^T L_31 v31
C32 = L3_V2(:,1)'*L3_dec_2*L3_V2(:,1);  %v32^T L_32 v32
C41 = L4_V1(:,1)'*L4_dec*L4_V1(:,1);    %v41^T L_41 v41
C42 = L4_V2(:,1)'*L4_dec_2*L4_V2(:,1);  %v42^T L_42 v42


L_fused = L_2 + C31*L3_dec+C32*L3_dec_2+C41*L4_dec+C42*L4_dec_2;



Method_Name(it) = "EigenVector_fused_Le";
[groups_fused,eigen_fused] = Eig_Lap(L_fused,cls_num);
[ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_fused);
it = it+1;
embedding_bank{cell_it}  = [eigen_fused];
results_bank{cell_it} = [groups_fused];
cell_it = cell_it + 1;
Method_Name(it) = "SVD_fused_Le";
[groups_svd,~] = Svd_Lap(L_fused,cls_num);
[ACC(it), ARI(it), F_SCORE(it), NMI(it), Purity(it)] = ClusterMeasure(gt,groups_svd);
it = it+1;


finale_resault_fused = [Method_Name',ACC',ARI',F_SCORE',NMI',Purity'];

xlswrite(save_path+file_name+'_result.xls', finale_resault_fused);
save(save_path+file_name+'_result.mat','data','gt','finale_resault_fused','embedding_bank', 'results_bank');

%%
%visual 
figure(1);
X_tsne = tsne(data);
scatter(X_tsne(:,1),X_tsne(:,2),'filled','cdata',gt);
saveas(gcf,save_path+file_name+'eigen.jpg');
saveas(gcf,save_path+file_name+'eigen');
for i = 1:2
    figure(i+1);
    X_tsne = tsne(embedding_bank{i});
    scatter(X_tsne(:,1),X_tsne(:,2),'filled','cdata',gt);
    saveas(gcf,save_path+file_name+'fused_'+string(i)+'.jpg');
    saveas(gcf,save_path+file_name+'fused_'+string(i));
end
% 
% for i = 1:4
%     V1_L23 = embedding_bank{(i-1)*3+1};
%     V1_L23_gc = embedding_bank{(i-1)*3+2};
%     V1_L24 = embedding_bank{(i-1)*3+3};
%     V1_L234 = embedding_bank{(i-1)*3+4};
%     V1_L234_gc = embedding_bank{(i-1)*3+5};
%     X = tsne(data);
%     XL23 = tsne(V1_L23);
%     XL23_gc = tsne(V1_L23_gc);
%     XL24 = tsne(V1_L24);
%     XL234 = tsne(V1_L234);
%     XL234_gc = tsne(V1_L234_gc);
%     %     XL23 = V1_L23;
%     %     XL24 = V1_L24;
%     %     XL234 = V1_L234;
%     figure(i);
%     subplot(611);
%     scatter(X(:,1),X(:,2),'filled','cdata',gt);
%     subplot(612);
%     scatter(XL23(:,1),XL23(:,2),'filled','cdata',gt);
%     subplot(613);
%     scatter(XL23_gc(:,1),XL23_gc(:,2),'filled','cdata',gt);
%     subplot(614);
%     scatter(XL24(:,1),XL24(:,2),'filled','cdata',gt);
%     subplot(615);
%     scatter(XL234(:,1),XL234(:,2),'filled','cdata',gt);
%     subplot(616);
%     scatter(XL234_gc(:,1),XL234_gc(:,2),'filled','cdata',gt);
%     
% end
end


function [L3_decom,L3_decom_2] = Kr_approximating(L3)

    [m,n] = size(L3);
    L3_decom = zeros(n,n);
    L3_decom_2 = zeros(n,n);
    for i = 1:n 
        temp = L3(:,i);
        [L3_decom(:,i),L3_decom_2(:,i),~] = NearestKroneckerProduct(temp,[n,1],[n,1]);
    end
end

function eigenV = eigen_vec(L,cls_num,max_flag)
    [ker_2,lamda_2] = eig(L);
    [m,~] = size(L);
    [ker_2,lamda_2] = cdf2rdf(ker_2,lamda_2);
    if max_flag == 0
        [max_lamda,lambda_index] = mink(diag(lamda_2),cls_num);
    else
        [max_lamda,lambda_index] = maxk(diag(lamda_2),cls_num);
    end
    ker_N_2 = ker_2(:,lambda_index);
    for i = 1:m
        kerNS(i,:) = ker_N_2(i,:) ./ norm(ker_N_2(i,:)+eps);
    end
    eigenV = kerNS;
end