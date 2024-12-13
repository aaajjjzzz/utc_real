function [L1, embedding] = computing4(A,num_cluster,opt_sim)
% Description: computing unfolded T with m^2 * m^2, where m is the number of samples.

% Input
%% > A: data matrix, with size m * n, where m is the number of samples and n is the number of features.

% Output
%% > unfolded T with m^2 * m^2, where m is the number of samples.
[m,n] = size(A);
disp(m)
H = pdist2(A,A,'euclidean');

disp('H4D tensor similarity')
T_tensor = zeros(m^2,m^2) + 1e-3;
i = 0;
j = 0;
k = 0;
l = 0;
for a = 1:m^2
    for b = a:m^2
        disp([a,b])
        i = mod(a,m);
        if i == 0
            i = m;
        end
        j = ceil(a/m);
        k = mod(b,m);
        if k == 0
            k = m;
        end
        l = ceil(b/m);
        disp([i,j,k,l])
        T_tensor(a,b) = (H(i,j)+H(k,l))/(H(i,k)+H(j,l)+1e-4);% inter-cluster/intra-class
    end
end
T_tensor = T_tensor' + T_tensor

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

