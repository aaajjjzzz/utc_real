function [groups,kerNS] = Eig_Lap_max(Lap,cls_num)
%Eig_LAP SC throgh VN in Lap's eigenvector directly
% Input£º
%       Lap      -      Lapician matrix 
%       cls_num  -      cluster number
% Output:
%       groups    -     clustering resualt
%       kerN      -     the first k-th eigenvector

% version 1.0 - 22/11/2021
% Written by Fei Qi
    m = size(Lap,1);
    [ker_2,lamda_2] = eig(Lap);
    [ker_2,lamda_2] = cdf2rdf(ker_2,lamda_2);
    [max_lamda,lambda_index] = maxk(diag(lamda_2),cls_num);
    ker_N_2 = ker_2(:,lambda_index);
    for i = 1:m
        kerNS(i,:) = ker_N_2(i,:) ./ norm(ker_N_2(i,:)+eps);
    end
    REPlic = 20; % Number of replications for KMeans
    MAXiter = 1000; % Maximum number of iterations for KMeans 
    groups = kmeans(kerNS,cls_num,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
end

