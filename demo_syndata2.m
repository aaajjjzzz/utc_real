clc;clear;

addpath(genpath('.\utils\methods_func'));

load('C:\科研\谱聚类\UTC\代码实现\UTC\dataset\syndata_2.mat');
[m,~] = size(gt);
num_cluster = length(unique(gt));
it =1;

%%
save_path = 'C:\科研\谱聚类\UTC\代码实现\UTC\result\';
file_name = 'syn_2';
opts.lambda1 = 1;
opts.lambda2 = 1;
opts.order4_sim_opt = 1;
opts.L2_opt = 1;
final_result = func_UTC(data, gt, file_name, save_path,opts);