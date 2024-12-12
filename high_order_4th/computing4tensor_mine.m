function [L1,embedding] = computing4tensor_mine(A,knn,num_cluster)
% Description: computing unfolded T with m^2 * m^2, where m is the number of samples.

% Input
%% > A: data matrix, with size m * n, where m is the number of samples and n is the number of features.
%% > knn: scalar, the number of nearest neighbors when sampling for tensor.

% Output
%% > unfolded T with m^2 * m^2, where m is the number of samples.
    [m,n]=size(A);
    % pairwise similarity
    H = zeros(m,m);
    for p = 1:m
        for q = 1:m
            H(p,q) = norm(A(p,:) - A(q,:));%n^2*m
        end
    end
    %% pair
    [dk, site] = tensor_clustering_new3(A,H,knn);% n*knn^3
    T = zeros(m,m,m,m)+1e-3;
    for i1 = 1:size(site,1)
        T(site(i1,1),site(i1,2),site(i1,3),site(i1,4)) = dk(i1,:);%T(i,k,j,l)
        T(site(i1,3),site(i1,4),site(i1,1),site(i1,2)) = dk(i1,:);%T(j,l,i,k)
        T(site(i1,3),site(i1,2),site(i1,1),site(i1,4)) = dk(i1,:);%T(j,k,i,l)%�����Ⱑ ����lth
        T(site(i1,1),site(i1,4),site(i1,3),site(i1,2)) = dk(i1,:);%T(i,l,j,k)%�����Ⱑ ����lth ������������⣬�������뵱Ȼ�ĺ�computing4tensorֱ�ӱ���һ���ˣ���ʵ��һ��
    end
    row = m*m;
    clear site dk;
    
    S1 = single(reshape(T,[row,row])); % unfolding tensor similarity
    clear T;
    vector = single(sum(S1,2));
    temp = single(ones(size(vector)))./sqrt(vector);
    clear vector;
    L1 = bsxfun(@times, S1, temp); % reduce computational time
    %% ȡk������ֵ(SNF)
    fprintf('start eig');% n^6
    [vec13,~] = eigs(double((L1+L1')/2),num_cluster);% ȡ������������
    Wall = cell(1,num_cluster);
    for i = 1:num_cluster
        Wall{i} = reshape(vec13(:,i),[m,m]);
    end
    embedding = SNF(Wall);
    
end

