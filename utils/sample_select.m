function [data_new, gt_new] = sample_select(data, gt, opt)
%SAMPLE_SELECT randomly select 
%   此处显示详细说明
    cluster_size = 30;
    cluster_num = 4;
    is_size_random = 0;
    if ~exist('opt', 'var')
        opt = [];
    end
    if isfield(opt, 'cluster_size');   cluster_size = opt.cluster_size;     end
    if isfield(opt, 'cluster_num');    cluster_num = opt.cluster_num;       end
    if isfield(opt, 'is_size_random'); is_size_random = opt.is_size_random; end
    cls_num = length(unique(gt));
    [m,n] = size(data);
    data_with_gt = [data,gt];
    sortrows(data_with_gt,n+1);
    data_new = [];
    gt_new = [];
    %% 
    Temp = randperm(cls_num);
    selected_cls = Temp(1:cluster_num); clear Temp;
    for i = 1:cluster_num
        cls_index = selected_cls(i);
        data_temp = data(gt==cls_index,:);
        gt_temp = gt(gt == cls_index);
        if is_size_random
            random_field = unidrnd(10)-5;          
        else
            random_field = 0;
        end
        subgroup_data = data_temp(1:cluster_size+random_field,:);
        subgroup_gt = gt_temp(1:cluster_size+random_field);
        data_new = [data_new; subgroup_data];
        gt_new = [gt_new; subgroup_gt];  
    end
    %%
    gt_label = sort(unique(gt_new));
    [gt_row_num,gt_col_num] = size(gt_label);
    for i = 1: gt_row_num
        gt_new(gt_new==gt_label(i)) = i;
    end
    %%
    [m,n] = size(data_new);
    disp('the size of new dataset is :'+string(m)+' and '+string(n));
    
end

