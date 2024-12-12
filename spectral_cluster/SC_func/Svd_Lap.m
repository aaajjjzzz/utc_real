function [groups,kerNS] = Svd_Lap(Lap, cls_num)
%SVD_LAP SC throgh VN in Lap's SVD 
% Input£º
%       Lap      -      Lapician matrix 
%       cls_num  -      cluster number
% Output:
%       groups    -     clustering resualt
%       kerN      -     vN

% version 1.0 - 27/07/2021
% Written by Fei Qi
m = size(Lap,1);
[uN,sN,vN] = svd(Lap);      
kerN = vN(:,m-cls_num+1:m);                    %%%%%%%%%%%%%%%%%%%%
for i = 1:m
    kerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
end
MAXiter = 1000; % Maximum number of iterations for KMeans
REPlic = 20; % Number of replications for KMeans
REPlic = 20; % Number of replications for KMeans
MAXiter = 1000; % Maximum number of iterations for KMeans 
groups = kmeans(kerNS,cls_num,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');

end

