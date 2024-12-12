function [S,D] = pair_similarity_singular(X)
%PAIR_SIMILARITY_SINGULAR �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    x1 = X(1,:);
    x2 = X(2,:);
    x3 = X(3,:);
    x4 = X(4,:);
    pic_M = [x1' x2' x3' x4'];
    [~, Singular, ~] = svd(pic_M);
    S = Singular(1,1)^2/norm(sum(Singular,1),'F')^2;
    D = Singular(4,4)^2/norm(sum(Singular,1),'F')^2;
end

