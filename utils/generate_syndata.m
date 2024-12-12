function [data,gt] = generate_syndata(m,n)
%GENERATE_SYNDATA 此处显示有关此函数的摘要
%   此处显示详细说明

A = zeros(m,n) + 1 * normrnd(0,1,m,n) + normrnd(0,0.5,m,n);
data(1:30,1:n) = 0.1 + normrnd(0,0.5,30,n);
data(31:60,1:n) = 0.5 + normrnd(0,0.5,30,n);
data(61:m,1:n) = 1 + normrnd(0,0.5,30,n);
data = data + A;
TL1 = zeros(1,m);
TL1(:,1:30) = 1;
TL1(:,31:60) = 2;
TL1(:,61:m) = 3;
TL1 = TL1';
gt = TL1;

end

