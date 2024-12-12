function [V_1,V_2,iter] = ASL24_um(L1,L2,cls_num,opts,lambda1)
% Solve the High order similarity(4+2) problem Multi-Clustering unified
% model
%
% min_{v_1,v_2} -{v_1}^T*L_1*{v_1} - {v_2}^T*L_2*{v_2}, 
%      s.t.{v_1}*_{kron}{v_1}={v_2}  {v_1^T}*{v_1}=I  {v_2^T}*{v_2}=I
%
% ---------------------------------------------
% Input£º
%       L1      -       n*n LapMatrix with pointwise similarity
%       L2      -       n^2*n*2 LapMatrix with pairwise limilarity
%       cls_num -       clustering number
%       opt     -        Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
% Output:
%       v_1     -       L_1's approximate eigenvecs n*k
%       v_2     -       L_2's approximate eigenvecs n^2*k
%       obj_rec -       objective record
%       iter    -       number of iterations
%       difY1_rec -     difY1_record  
%       difY2_rec -     difY2_record
% version 1.2 - 18/08/2021
% Written by Fei Qi
%%
%para
tol = 1e-8; 
max_iter = 500;
rho = 1.1;
mu = 1e-4;
%max_mu = 1e10;
max_mu = 1e-2;
DEBUG = 0;
alpha = 1e-13;
L1 = L1/norm(L1(:));
%L2 = 1*L2/norm(L2(:));
L2 = lambda1*L2/norm(L2(:));
%cls_num=1;
if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
%if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end
if isfield(opts, 'alpha');       alpha = opts.alpha;          end

%%
%initial
[n_1,n_1] =  size(L1);
[n_2,n_2] = size(L2);
%cls_num = 3;
m_1 = cls_num;
iter = 1;
V_1 = abs(randn(n_1,cls_num));
V_2 = abs(randn(n_2,cls_num));
Y1 = 0.1*ones(n_2,cls_num);
Y2 = 0.1*ones(cls_num,cls_num);
a = 1e-3;
is_test = 0;

V1_k = V_1;
V2_k = V_2;
Y1_k = Y1;
Y2_k = Y2;

%%
while true
    V1_t = V1_k;
    V2_t = V2_k;
    Y1_t = Y1_k;
    Y2_t = Y2_k;
    if iter==1
        %         [u_1,s_1,vv_1] = svd(full(eye(n_1)-L1));
        %         V1_k = vv_1(:,n_1-cls_num+1:n_1);
        %         V2_k = kron_col(V1_k);
        
        %
        %         [u_1,s_1,vv_1] = svd(full(L1));
        %         V1_k = vv_1(:,1:cls_num);
        %         V2_k = kron_col(V1_k);
        
        
        [ker_2,lamda_2] = eig(L1);
        [ker_2,lamda_2] = cdf2rdf(ker_2,lamda_2);
        [max_lamda,lambda_index] = maxk(diag(lamda_2),cls_num);
        ker_N_2 = ker_2(:,lambda_index);
%         for i = 1:m
%             kerNS(i,:) = ker_N_2(i,:) ./ norm(ker_N_2(i,:)+eps);
%         end
        V1_k = ker_N_2;
        V2_k = kron_col(V1_k);

    else
        %Update V1
        V1_k = update_V1_L24(V1_k,V2_k,L1,Y1_k,Y2_k,mu,alpha,1);
        %f_1 = obj_f(V1_k,V2_k,L1,L2,mu,Y1_k,Y2_k);
        %normalization
        
%         for i = 1:cls_num
%             V1_k(:,i) = V1_k(:,i)/norm(V1_k(:,i));
%         end
        
        %Update V2
        %V2_k = update_V2_gradientdescent(V1_k,V2_k,L2,Y1_k,mu,alpha,is_test);
        %V2_k = inv(mu*eye(n_2)+2*L2)*(Y1_k+mu*(kron_col(V1_k)));
        A = (mu*eye(n_2)-2*L2);
        B = (Y1_k+mu*(kron_col(V1_k)));
        for B_n = 1:size(B,2)
            V2_k(:,B_n) = gmres(A,B(:,B_n));
        end
        
        
        %V2_k = inv(mu*eye(n_2)-2*L2)*(Y1_k+mu*(kron_col(V1_k)));
        %f_2 = obj_f(V1_k,V2_k,L1,L2,mu,Y1_k,Y2_k);
        %Update Y1
        Y1_k = Y1_k + mu*(kron_col(V1_k)-V2_k);
        %Update Y2
        Y2_k = Y2_k + mu*(V1_k'*V1_k-eye(m_1));
    end
    
    chg_V1 = max(max(abs(V1_t-V1_k)));
    chg_V2 = max(max(abs(V2_t-V2_k)));
    chg_Y1 = max(max(abs(Y1_t-Y1_k)));
    chg_Y2 = max(max(Y2_t-Y2_k));
    chg = max([chg_V1 chg_V2 chg_Y1 chg_Y2]);
        iter = iter+1;
    if chg < tol || iter>max_iter
        break;
    end
    if DEBUG
        if iter==1 || mod(iter, 10)==0
            obj = obj_f(V1_k,V2_k,L1,L2,mu,Y1_k,Y2_k);
            disp(['iter ' num2str(iter) ', obj=' num2str(obj) ', mu=' num2str(mu)]);                  
        end
    end
%     LL1 = -trace(V1_k'*L1*V1_k);
%     LL2 = -trace(V2_k'*L2*V2_k);
%     obj_rec(iter,:) = [LL1,LL2,LL1+LL2];
%     difY1_rec(iter) = norm(kron_col(V1_k)-V2_k);
%     difY2_rec(iter) = norm(V1_k'*V1_k-eye(cls_num));
    mu = min(rho*mu,max_mu);   
    %mu = min(rho*mu,max_mu);   
    %mu = max(rho*mu,min_mu);   
end
    V_1 = V1_k;
    V_2 = V2_k;
end

function Z = obj_f(V1,V2,L1,L2,mu,Y1,Y2)
    [m1,n1] = size(V1);
    [m2,n2] = size(V2);
    z1 = trace(V1'*L1*V1);
    z2 = trace(V2'*L2*V2);
    z3 = trace(Y1'*(kron_col(V1)-V2));
    z4 = trace(Y2'*(V1'*V1-eye(n1)));
    z6 = mu/2* (norm(kron_col(V1)-V2,'fro')*norm(kron_col(V1)-V2,'fro') ...
        +norm(V1'*V1-eye(n1),'fro')*norm(V1'*V1-eye(n1),'fro') );
    %Z = -z1-z2+z3+z4+z6;
    Z = -z1-z2;
end

function VV1 = kron_col(V1)
    [m,n] = size(V1);
    VV1 = zeros(m*m,n);
    for i = 1:n
        v1i = V1(:,i);
        VV1(:,i) = kron(v1i,v1i);
    end
end
