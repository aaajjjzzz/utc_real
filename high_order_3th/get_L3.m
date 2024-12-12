function [L3] = get_L3(data,knn,num_cluster,L_3_style)
%GET L1 embedding
%
%
% ---------------------------------------------
% Input£º
%       data        -       data m*n
%       knn         -       nearest neighbour num
%       num_cluster -       cluster_num
%       L_3_style   -       3-order similarity approach 
%                              - 1 stands for the cosine
%                              - 3 stands for the gravity centroid 
% Output:
%       L3           -       L3_spread;
%       embedding    -       L3's embedding
% version 1.1 - 02/09/2021
% Written by Fei Qi

    [m,n] = size(data);
    S = pdist2(data,data,'euclidean');
    [L_3_dk, L_3_dk_index] = computing3tensor(data,S,knn*knn,L_3_style);  %1 old style; 3 for gravity centroids
    
    disp('HD 3-order tensor similarity') 
    row_1 = zeros(1, size(L_3_dk_index,1));
    column_1 = zeros(1, size(L_3_dk_index,1));
    
    
    for i1 = 1:size(L_3_dk_index,1)
        row_1(i1) = m*(L_3_dk_index(i1,3)-1)+L_3_dk_index(i1,1);%
        column_1(i1) = L_3_dk_index(i1,2);
    end
    
    row_all = single(row_1);
    column_all = single(column_1);
    clear row_1 column_1   
    dk_all = L_3_dk;
    clear dk
    [uni_ind, ind, ~] = unique([row_all',column_all'], 'rows');
    dk_uni = dk_all(ind);
    T_tensor = accumarray(uni_ind,dk_uni',[m^2,m],[],[],1);
    vector_1 = sqrt(kron(sum(T_tensor,1),sum(T_tensor,1)));
    vector_2 = (sum(T_tensor,1));
    temp_1 = (ones(size(vector_1)))./sqrt(vector_1);
    temp_2 = (ones(size(vector_2)))./sqrt(vector_2);
    clear vector_1 vector_2;
    num_diag_1 = [1:m^2];
    num_diag_2 = [1:m];
    D_inv_1 = sparse(num_diag_1, num_diag_1, temp_1);
    D_inv_2 = sparse(num_diag_2, num_diag_2, temp_2);
    L3 = D_inv_1 * T_tensor * D_inv_2;
    %clear T_tensor site dk;
    % clear site dk;
    %
    
    %
    % fprintf('start eig');% n^6
    % %
    
    
    %{
    fprintf('start eig');% n^6
    %opts.tol = 1e-2;
    [vec13,eigva1] = eigs(double(L1+L1')/2,num_cluster);%
    embedding = zeros(m,m);
    for i = 1:num_cluster
        embedding = embedding + reshape(vec13(:,i),[m,m]);
        %matrix_snf_2 = matrix_snf_2/cluster;
    end
    %}
end

