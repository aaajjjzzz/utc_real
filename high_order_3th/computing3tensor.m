function [dk1,dk_index1] = computing3tensor(data,S,knn,opt_sim)
% Get 3-order tensor by data through cosine
%
%
% ---------------------------------------------
% Input：
%       data    -       data m*n
%       S       -       Similar Matrix of data
%       knn     -       nearest neighbour num
%       opt_sim -       Similarity option
% Output:
%       dk      -       similarity of 3 points;
%       dk_index    -       index of the similarity
% version 1.1 - 02/09/2021
% Written by Fei Qi
    [n,~] = size(data);
    dk = cell(n,1);
    dk_index = cell(n,1);
    dis = pdist2(data,data,'euclidean');%这已经算过一遍了还算。——lth
    dis_max = max(max(dis));
    if opt_sim == 1 % use cos_similarity
        for num = 1:n
            dk{num} = [];
            dk_index{num} = [];
            [~,id] = sort(S(num,:));
            pair = pair_com(id(2:2*knn+1),2);
            [p_row, ~] = size(pair);
            disp(1)
            for i1 = 1:p_row 
                d_i = data(pair(i1,1),:)-data(num,:);
                d_j = data(pair(i1,2),:)-data(num,:);%这里与后面41行，44行，
                if d_i*d_j' <0
                    d_i = -d_i;
                end
                if dis(pair(i1,1),num)==0 || dis(pair(i1,2),num)==0 
                    d1 = 1-dis(pair(i1,1),num)/dis_max;
                    d2 = 1-dis(pair(i1,2),num)/dis_max;
                    S_ijk = max([d1,d2]);
                    
                else
                    S_ijk = 1-pdist([d_i;d_j],'cosine');
                end
                dk{num} = [dk{num}; S_ijk];
                dk_index{num} = [dk_index{num};[pair(i1,1),pair(i1,2),num]];

            end
        end
    elseif opt_sim == 2 % use cos_similarity_full
        for num = 1:n
            dk{num} = [];
            dk_index{num} = [];
            for i = 1:n
                for j = 1:n
                    d_i = data(i,:)-data(num,:);
                    d_j = data(j,:)-data(num,:);
                    S_ijk = squareform(1-pdist([d_i;d_j],'cosine'));
                    dk{num} = [dk{num}; S_ijk];
                    dk_index{num} = [dk_index;[i,j,num]];%这里有问题
                end
            end
        end
    elseif opt_sim == 3 % use gravity centroid 
        for num = 1:n
            dk{num} = [];
            dk_index{num} = [];
            [~,id] = sort(S(num,:));
            pair = pair_com(id(2:2*knn+1),2);
            [p_row, ~] = size(pair);
            for i1 = 1:p_row 
                x_i = data(num,:);
                x_j = data(pair(i1,1),:);
                x_k = data(pair(i1,2),:);
                d_ijk = triadic_dis_gravity(x_i,x_j,x_k);
                %S_ijk = 1-pdist([d_i;d_j],'cosine');
                dk{num} = [dk{num}; d_ijk];
                dk_index{num} = [dk_index{num};[pair(i1,1),pair(i1,2),num]];
            end
        end
    elseif opt_sim == 4 % use tri_S for sim
        for num = 1:n
            dk{num} = [];
            dk_index{num} = [];
            [~,id] = sort(S(num,:));
            pair = pair_com(id(2:2*knn+1),2);
            [p_row, ~] = size(pair);
            for i1 = 1:p_row
                x_i = data(num,:);
                x_j = data(pair(i1,1),:);
                x_k = data(pair(i1,2),:);
                d_ij = norm(x_i-x_j,'fro');
                d_jk = norm(x_j-x_k,'fro');
                d_ik = norm(x_i-x_k,'fro');
                p = (d_ij+d_jk+d_ik)/2;
                if p*(p-d_ij)*(p-d_ik)*(p-d_jk)<0
                    continue;
                end
                d_ijk = sqrt(p*(p-d_ij)*(p-d_ik)*(p-d_jk));
                
                %S_ijk = 1-pdist([d_i;d_j],'cosine');
                dk{num} = [dk{num}; d_ijk];
                dk_index{num} = [dk_index{num};[pair(i1,1),pair(i1,2),num]];
            end
        end
    end
    dk1 = dk{1};
    dk_index1 = dk_index{1};
    for kk = 2:n
        dk1 = [dk1;dk{kk}];   %四个点之间相似度
        dk_index1 = [dk_index1; dk_index{kk}]; %对应的索引
    end
    dk1 = double(dk1);
    if opt_sim == 3
        max_dis = max(dk1);
        dk1 = 1 - dk1/max_dis;
    elseif opt_sim == 4
        [m,~] = size(dk1);
        for i = 1:m
            if dk1(i) > 1e-5
                dk1(i) = 1/dk1(i);
            end
        end
        max_dis = max(dk1);
        for i = 1:m
            if dk1(i) ~= 0
                dk1(i) = dk1(i)/max_dis;
            else
                dk1(i) = 1;
            end
        end
    end
    
end


function d_ij = pairwise(x_i,x_j)
    d_ij = sqrt(norm(x_i-x_j,2));
end

function d_ijk = triadic_dis_gravity(x_i,x_j,x_k)
    g_ijk = (x_i+x_j+x_k)/3;
    d_ijk = (pairwise(x_i,g_ijk)+pairwise(x_j,g_ijk)+pairwise(x_k,g_ijk))/3;
end
