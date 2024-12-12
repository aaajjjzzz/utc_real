function [Sk, hyperedge_ind] = pair2pair_vector(X,S)
%PAIR2PAIR_VECTOR 此处显示有关此函数的摘要
% 寻找每个点的k近邻，构建超边（i,j,k,l）
    k = 4;
    knn = 10;
    [n,~] = size(X);
    dk = zeros(n,1);
    hyperedge_num = k*nchoosek(n,2);
    v_per_he_num = fix(hyperedge_num/n);
    ad={};
    for i = 1:n
        [~, knn_id] = sort(S(i,:));
%         id_index = knn_id(1:min([v_per_he_num n]));
%         ad{i} = nchoosek(id_index,4);
        id_index = nchoosek(knn_id,4);
        ad{i} = id_index(1:v_per_he_num,:);
    end
    index = ad{1};
    for i = 2:n
        index = [index;ad{i}];
    end
    dk_index = sort(index, 2);
    [hyperedge_ind, ind, ~] = unique(dk_index, 'rows');
    he_num = size(hyperedge_ind,1);
    Sk=zeros(he_num,1);
    parfor i = 1:he_num
        x1 = X(hyperedge_ind(i,1),:);
        x2 = X(hyperedge_ind(i,2),:);
        x3 = X(hyperedge_ind(i,3),:);
        x4 = X(hyperedge_ind(i,4),:);
        Sk(i) = pair_similarity_singular([x1; x2; x3; x4]);
    end
    
end

%    kk = 4;
%    op2 = 0;
%    [n,~] = size(X);
%    ac = 34220;
%    dk = zeros(ac,1);
%    ad = zeros(size(dk,1),kk);
%     for i1 = 1:n
%         for j1 = i1+1:n
%             for i2 = j1+1:n
%                 for j2 = i2+1:n
%                     op2 = op2 + 1;
%                     %dk(op2) = (S(i1,j1)+S(i1,i2)+S(j1,i2))/3;
%                     [dk(op2),~] = pair_similarity_singular([X(i1,:); X(j1,:); X(i2,:); X(j2,:)]);
%                     ad(op2,:) = [i1,j1,i2,j2];
%                 end
%             end
%         end
%     end