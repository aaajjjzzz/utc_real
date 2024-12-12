function V2_new = update_V2_gradientdescent(V1,V2,L_2,Y1,mu,alpha,is_test)
%UPDATE_V2_GD 此处显示有关此函数的摘要
%   update
    min_dif = 1e-4;
    iter= 0;
    t_V2 =V2;
    [m1 n1] = size(V1);
    [m2 n2] = size(V2);

%%
%use gd to update v2
    while true
        if is_test ==1
            f_g = zeros(m2,n2);
            for i=1:m2
                for j = 1:n2
                    delta_x = zeros(m2,n2);
                    delta_x(i,j) = 1e-5;
                    delta_f = obj_f_V2(V1,t_V2+delta_x,L_2,Y1,mu)- obj_f_V2(V1,t_V2,L_2,Y1,mu);
                    f_g(i,j) = delta_f/delta_x(i,j);
                    
                end
            end
            f_g_partial = L_2*t_V2 + L_2'*t_V2...
                -mu*(kron_col(V1)-t_V2+Y1/mu);
            df_fg12 = norm(f_g_partial-f_g,1);
            
            t_V2 = t_V2 - alpha*f_g;  
        else
            f_g = L_2*t_V2 + L_2'*t_V2...               %L_2*V_2^k+L_2^T*V_2^k
                -mu*(kron_col(V1)-t_V2+Y1/mu);          %-\mu()
            t_V2 = t_V2 - alpha*f_g;  
        end
        iter = iter+1;
        if norm(f_g) < min_dif || iter > 1e4
            break;
        end
        if is_test
            if iter==1 || mod(iter,10) == 0
                obj = obj_f_V2(V1,t_V2,L_2,Y1,mu);
                disp(['updating v2: iter = ' num2str(iter) ',obj=' num2str(obj) ', df_fg12 =' num2str(df_fg12)]);
            end
        end
    end
     V2_new = t_V2;        
end


function Z = obj_f_V2(V1,V2,L_2,Y1,mu)
    [m1,n1] = size(V1);
    [m2,n2] = size(V2);
    %z1 = trace(V1'*L_1*V1);
    z2 = trace(V2'*L_2*V2);
    z3 = trace(Y1'*(kron_col(V1)-V2));
    %z4 = trace(Y2'*(V1'*V1-eye(m1)));
    %z5 = trace(Y3'*(V2'*V2-eye(n2)));
    %============================================================================
%     z6 = mu/2* (norm(kron_col(V1)-V2,'fro')*norm(kron_col(V1)-V2,'fro') ...
%         +norm(V2'*V2-eye(n2),'fro')*norm(V2'*V2-eye(n2),'fro') );
    z6 = (mu/2)* (norm(kron_col(V1)-V2,'fro')^2);
    %============================================================================
    Z = z2+z3+z6;
end


function Z2 = obj_f2(V1,V2,Y1,mu)
    z3 = trace(Y1'*(kron_col(V1)-V2));
    z6 = (mu/2)* (norm(kron_col(V1)-V2,'fro')^2);
    Z2 = z3+z6;
end