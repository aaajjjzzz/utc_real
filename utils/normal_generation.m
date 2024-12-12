function [sample,TL1, cluster] = normal_generation(d,n)
%NORMAL_GENERATION 此处显示有关此函数的摘要
%   此处显示详细说明
% generating Gaussian distribution sample
%d = 2; % feature dimension
disp('loading normal distribution data, cluster_number=2');
cluster=2;

% Sigma matrix
Sigma_matrix = eye(d);
% Sigma_matrix(1:2:end, 2:2:end) = 0.98;
% Sigma_matrix(2:2:end,1:2:end) = 0.98;
for i = 1:fix(d/2)
    Sigma_matrix(2*i-1,2*i) = 0.98;
    Sigma_matrix(2*i,2*i-1) = 0.98;
end
% for N1:
mu_1 = zeros(1, d);
sigma_square_1 = 0.5;
Sigma_1 = sigma_square_1* Sigma_matrix;
% for N2:
mu_2 = zeros(1, d);
mu_2(1:2:end) = 1;
mu_2(2:2:end) = -1;
%mu_2 = mu_2 * 1.6;
mu_2 = mu_2 * 1.3;
sigma_square_2 = 2;
Sigma_2 = sigma_square_2* Sigma_matrix;

%figure;imagesc(Sigma_matrix);colorbar;

r_1 = mvnrnd(mu_1,Sigma_1,n/2);
r_2 = mvnrnd(mu_2,Sigma_2,n/2);

%plot3(r_1(:,1),r_1(:,2),r_1(:,3),'r+');hold on;
%plot3(r_2(:,1),r_2(:,2),r_1(:,3),'b+');


sample = [r_1;r_2];
% figure
% plot(r_1(:,1),r_1(:,2),'ko');hold on;
% plot(r_2(:,1),r_2(:,2),'b+');
% 
% 
% % xlim([-2 4]);
% % ylim([-4 3]);
% %l = title('Two Gaussian distributions in case.(b)','Fontsize',19);
% la_1 = xlabel({'$x^{(1)}$'},'Fontsize',18);
% la_2 = ylabel({'$x^{(2)}$'},'Fontsize',18);
% %set(l,'Interpreter','latex');
% set(la_1,'Interpreter','latex');
% set(la_2,'Interpreter','latex');
% 
% sigma_square_1
% sigma_square_2

TL1 = [ones(n/2,1);2*ones(n/2,1)];
end

