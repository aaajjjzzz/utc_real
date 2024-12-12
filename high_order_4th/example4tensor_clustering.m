function nmi_value= example4tensor_clustering

[h2,TL1] = syn_orth_data;
%[h2,TL1] = syn_gaussian_data;
level =3; k=4;
%%% By 1d spetral method;
% h2z = ssc_relaxed(h2,0.1);
h2= (h2-repmat(mean(h2),size(h2,1),1))./(repmat(std(h2),size(h2,1),1));
num_fea=10;sd = 2; m=20;
h2 = [1+sd*randn(m,num_fea);
      -1+sd*randn(m,num_fea);]
 truelabel = repmat([1; 2],1,m);%4*10µÄ¾ØÕó
 TL = truelabel'; TL1 = TL(:);TL = TL(:);

[h3,id1,S3] = SpectralClustering3(h2',2);
nmi_value(1) = nmi(id1,TL1);
subplot(2,2,1); imagesc(h3); title(strcat('NMI = ',num2str(nmi_value(1))));
%%% By 2d spetral method;
[result,id21,S21]= tensor_spetral_method_v3(h2,k,0);
nmi_value(2) = nmi(id21,TL1);
subplot(2,2,2); imagesc(result); title(strcat('NMI = ',num2str(nmi_value(2))));
[result,id22,S22]= tensor_spetral_method_v3(h2,k,1);
nmi_value(3) = nmi(id22,TL1);
subplot(2,2,3); imagesc(result); title(strcat('NMI = ',num2str(nmi_value(3))));
[result,id23,S23]= tensor_spetral_method_v3(h2,k,2);
nmi_value(4) = nmi(id23,TL1);
subplot(2,2,4); imagesc(result); title(strcat('NMI = ',num2str(nmi_value(4))));
%%% By 1d+2d spetral method;

[id3] = SpectralClustering(exp(0.1*(normalize(S3)+normalize(S22))),k);
[~,idx3] = sort(id3);
nmi_value(5) = nmi(id3,TL1);
subplot(2,2,4); imagesc(h2(idx3,:)); title(strcat('NMI = ',num2str(nmi_value(5))));

[id4] = SpectralClustering(normalize(S3)+normalize(S23),k);
[~,idx4] = sort(id4);
nmi_value(6) = nmi(id4,TL1);
subplot(2,2,4); imagesc(h2(idx3,:)); title(strcat('NMI = ',num2str(nmi_value(6))));
knn = 5;
num_cluster = 2;MAXiter = 1000; % Maximum number of iterations for KMeans 
REPlic = 20; % Number of replications for KMeans
L2 = constructW(h2);
 [L4, embedding] = computing4tensor(h2, knn, num_cluster);
v1 = compute_embedding_vector(L2,L4);
id7 = kmeans(v1,num_cluster,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
nmi_value(7) = nmi(id7,TL1);
[~,y_pred,~] = SpectralClustering5(embedding, num_cluster); %
[ACC, ARI, F_SCORE, NMI, Purity] = ClusterMeasure(TL1, y_pred);


function [h2,TL1] = syn_orth_data
m =40;
l = 0:10:m;
h = 0.01*rand+.05*randn(m);
 for i=1:length(l)-1
     h(l(i)+1:l(i+1),l(i)+1:l(i+1)) = 0.5+.5*randn(10);
 end
 %h = [h; 0.01*randn(40,size(h,2))]; %h = [h, randn(size(h,1),40)];
 h1 = h;
 truelabel = repmat([1; 2; 3; 4],1,m/4);%4*10µÄ¾ØÕó
 TL = truelabel'; TL1 = TL(:);TL = TL(:);
order2=randperm(size(h1,1));
order2 = 1:length(order2);
h2 = h1(order2,:);
TL1 =TL(order2);
%h2 = h1;

 function [h2,TL1] = syn_gaussian_data
     sd= 2.5; num_fea=10;
h2 = [1+sd*randn(10,num_fea),5+sd*randn(10,num_fea);
    1+sd*randn(10,num_fea),1+sd*randn(10,num_fea);
    5+sd*randn(10,num_fea),1+sd*randn(10,num_fea);
    5+sd*randn(10,num_fea),5+sd*randn(10,num_fea);];
TL1 = repmat([1 2 3 4],10,1);
TL1 = TL1(:);

    
