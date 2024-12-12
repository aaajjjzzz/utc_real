clc;clear;

load('C:\科研\谱聚类\UTC\代码实现\UTC_real\dataset\DriveFave.mat');
[m,~] = size(gt);
num_cluster = length(unique(gt));
it =1;

%%
save_path = 'C:\科研\谱聚类\UTC\代码实现\UTC_real\result\';
file_name = 'DriveFave';
opts.lambda1 = 1;
opts.lambda2 = 1;
opts.order4_sim_opt = 1;
opts.L2_opt = 0.75;
final_result = func_UTC(data, gt, file_name, save_path,opts);