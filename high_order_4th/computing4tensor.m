function [L1, embedding] = computing4tensor(A,knn, num_cluster,opt_sim)
% Description: computing unfolded T with m^2 * m^2, where m is the number of samples.

% Input
%% > A: data matrix, with size m * n, where m is the number of samples and n is the number of features.
%% > knn: scalar, the number of nearest neighbors when sampling for tensor.

% Output
%% > unfolded T with m^2 * m^2, where m is the number of samples.
[m,n] = size(A);
H = pdist2(A,A,'euclidean');


[dk, site] = tensor_clustering_tcyb(A,H,knn*2,opt_sim);% n*knn^3; A:input, n_sample * n_feature; H: distance matrix; knn: number of knn
disp('HD tensor similarity')

row_1 = zeros(1, size(site,1));
row_2 = zeros(1, size(site,1));
row_3 = zeros(1, size(site,1));
row_4 = zeros(1, size(site,1));
column_1 = zeros(1, size(site,1));
column_2 = zeros(1, size(site,1));
column_3 = zeros(1, size(site,1));
column_4 = zeros(1, size(site,1));

for i1 = 1:size(site,1)
    row_1(i1) = m*(site(i1,2)-1)+site(i1,1);
    column_1(i1) = m*(site(i1,4)-1)+site(i1,3);
end
for i1 = 1:size(site,1)
    row_2(i1) = m*(site(i1,4)-1)+site(i1,3);
    column_2(i1) = m*(site(i1,2)-1)+site(i1,1);
end
for i1 = 1:size(site,1)
    
    row_3(i1) = m*(site(i1,2)-1)+site(i1,3);
    column_3(i1) = m*(site(i1,4)-1)+site(i1,1);
end
for i1 = 1:size(site,1)
  
    row_4(i1) = m*(site(i1,4)-1)+site(i1,1);
    column_4(i1) = m*(site(i1,2)-1)+site(i1,3);
end

row_all = single([row_1, row_2, row_3, row_4]);
column_all = single([column_1, column_2, column_3, column_4]);
clear row_1 row_2 row_3 row_4 column_1  column_2 column_3 column_4
dk_all = [dk;dk;dk;dk];
clear dk
[uni_ind, ind, ~] = unique([row_all',column_all'], 'rows');
dk_uni = dk_all(ind);

%S_all = sparse(row_all, column_all, dk_all, m^2, m^2);
T_tensor = accumarray(uni_ind,dk_uni',[m^2,m^2],[],[],1);

vector = (sum(T_tensor,2));
temp = (ones(size(vector)))./sqrt(vector);
clear vector;
num_diag = [1:m^2];
D_inv = sparse(num_diag, num_diag, temp);
L1 = D_inv * T_tensor * D_inv;
%clear T_tensor site dk;
% clear site dk;
% 

% 
% fprintf('start eig');% n^6
% % 
fprintf('start eig');% n^6
%opts.tol = 1e-2;
[vec13,eigva1] = eigs(double(L1+L1')/2,num_cluster);%
embedding = zeros(m,m);
for i = 1:num_cluster
    embedding = embedding + reshape(vec13(:,i),[m,m]);
    %matrix_snf_2 = matrix_snf_2/cluster;
end

end

