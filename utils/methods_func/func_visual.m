function embedding_num = func_visual(data,gt,embedding_bank, dataset_name, save_path)
%FUNC_VISUAL visual the embedding though t-sne
%   此处显示详细说明

[m,n] = size(embedding_bank);
figure(1);
X_tsne = tsne(data);
scatter(X_tsne(:,1),X_tsne(:,2),'filled','cdata',gt);
saveas(gcf,save_path+string(dataset_name)+'_raw_data.jpg');
saveas(gcf,save_path+string(dataset_name)+'raw_data');
result_name = {'_L23_cos','_L23_gc','_L24','_L234_cos','_L234_gc'};

for i = 1:n/6

%     XL23 = V1_L23;
%     XL24 = V1_L24;
%     XL234 = V1_L234;
    for j = 1:5
        figure(j+1)
        X_tsne = tsne(embedding_bank{(i-1)*5+j+2});
        scatter(X_tsne(:,1),X_tsne(:,2),'filled','cdata',gt);
        saveas(gcf,save_path+string(dataset_name)+string(cell2mat(result_name(j)))+'.jpg');
        saveas(gcf,save_path+string(dataset_name)+string(cell2mat(result_name(j))));
    end
end
embedding_num = n;

end

