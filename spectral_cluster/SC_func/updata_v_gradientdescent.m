function [V_new,obj,iter] = updata_v_gradientdescent(V,L,Y,mu,alpha,DEBUG)
% update v through gradient_descent method
%
%
% ---------------------------------------------
% Input£º
%       V       -       the matrix need to update
%       L       -       the Lapician Matrix
%       Y       -       Langurage Var 
%       mu      -       F-norm cons
%       alpha   -       step 
%       DEBUG   -       DEBUG mod switch
% Output:
%       V_new   -       update V
%       obj     -       objective function value
%       iter    -       number of iterations
% version 1.0 - 27/07/2021
% Written by Fei Qi
    
%% para
    iter =0;
    V_k = V;
    V_t = V_k;
    [m, cls_num] = size(V_t);

%% Update Loop
    while true
        if DEBUG == true
            f_g = zeros(m,cls_num);
            for i=1:m
                for j= 1:cls_num
                    delta_vij = zeros(m,cls_num);
                    delta_vij(i,j) = 1e-4;
                    delta_fij = obj_f_v(V_k+delta_vij,L,Y,mu)-obj_f_v(V_k,L,Y,mu);
                    f_g(i,j) = delta_fij/delta_vij(i,j);
                end
            end
            f_g2 = 2*L*V_k + 2*mu*V_k*(V_k'*V_k-eye(cls_num)+Y/mu);
            V_k = V_k - alpha*f_g;
        else
            %f_g = -2*L*v_k + 2*mu*v_k*(v_k'*v_k-eye(cls_num)+Y/mu);
            f_g = 2*L*V_k + 2*mu*V_k*(V_k'*V_k-eye(cls_num)+Y/mu);
            V_k = V_k - alpha*f_g;
        end
        iter = iter+1;
        if norm(f_g) < 1e-5 || iter>1e4
            break;
        end
        if DEBUG == true
            if iter==1 || mod(iter, 50)==0
                 obj = obj_f_v(V_k,L,Y,mu);
                disp(['v iter ' num2str(iter) ', obj=' num2str(obj) ]);                  
            end           
        end
    end
    V_new = V_k;
end

function L_V = obj_f_v(V,L,Y,mu)
    m = size(Y,2);
    z1 = trace(V'*L*V);
    z2 = trace(Y'*(V'*V-eye(m)));
    z3 = mu/2*norm(V'*V-eye(m),'fro')*norm(V'*V-eye(m),'fro');
    L_V = z1+z2+z3;
end
