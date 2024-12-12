clc;clear;

load('F:\Work\Tensor Clustering\Tensor Spectral Clustering\high_order\affinies_orders_v2_0815\UTC\dataset\lymphoma.mat');
[m,~] = size(gt);
num_cluster = length(unique(gt));
it =1;

%%
save_path = 'F:\Work\Tensor Clustering\Tensor Spectral Clustering\high_order\affinies_orders_v2_0815\UTC\Result\';
file_name = 'lymphoma';
opts.lambda1 = 1;
opts.lambda2 = 1;
opts.order4_sim_opt = 1;
opts.L2_opt = 1;
final_result = func_UTC(data, gt, file_name, save_path,opts);