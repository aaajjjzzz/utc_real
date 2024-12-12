function V_new = update_V1_L24(V1,V2,L_1,Y1,Y2,mu,alpha,is_test)
%UPDATA_V1_GD 
%   update 
    min_dif = 1e-2;
    iter =0;
    t_V1 = V1;
    [m1,n1] = size(V1);
    [m2,n2] = size(V2);
%% use gd to 
    while true
        if is_test-1
            f_g = zeros(m1,n1);
            for i = 1:m1
                for j = 1:n1
                    delta_x = zeros(m1,n1);
                    delta_x(i,j) = 1e-8;
                    delta_f = obj_f_V1(t_V1+delta_x,V2,L_1,Y1,Y2,mu) - obj_f_V1(t_V1,V2,L_1,Y1,Y2,mu);
                    f_g(i,j) = delta_f / delta_x(i,j);
                end
            end
            %=================================================================
%             f_g_partial = -2*L_1*t_V1+...
%                 2*mu*t_V1*(t_V1'*t_V1-eye(n1)+Y2/mu)...
%                 +mu*( (kron(eye(n1),t_V1)+kron(t_V1,eye(n1)))' * (kron_col(t_V1)-V2+Y1/mu));
            f_g_partial = -2*L_1*t_V1+...
                2*mu*t_V1*(t_V1'*t_V1-eye(n1)+Y2/mu)...
                +mu*(partial_kron_col(t_V1,V2,Y1,mu));
            %==================================================================
            %df_fg12 = norm(f_g_partial-f_g,1);
            t_V1 = t_V1 - alpha*f_g;  
        else    
           %=================================================================
%             f_g_partial = -2*L_1*t_V1+...
%                 2*mu*t_V1*(t_V1'*t_V1-eye(n1)+Y2/mu)...
%                 +mu*( (kron(eye(n1),t_V1)+kron(t_V1,eye(n1)))' * (kron_col(t_V1)-V2+Y1/mu));
            f_g = -2*L_1*t_V1+...
                2*mu*t_V1*(t_V1'*t_V1-eye(n1)+Y2/mu)...
                +mu*(partial_kron_col(t_V1,V2,Y1,mu) );
            %==================================================================
            t_V1 = t_V1 - alpha*f_g;   
        end
        iter = iter+1;
        if norm(f_g) < min_dif || iter > 1e4
            break;
        end
        if is_test
            if iter==1 || mod(iter,10) == 0
                obj = obj_f_V1(t_V1,V2,L_1,Y1,Y2,mu);
                %disp(['updating v1: iter = ' num2str(iter) ',obj=' num2str(obj) ', df_fg12 =' num2str(df_fg12)]);
                disp(['updating v1: iter = ' num2str(iter) ',obj=' num2str(obj)]);
            end
        end
    end
     V_new = t_V1;
end


function Z = obj_f_V1(V1,V2,L_1,Y1,Y2,mu)
    [m1,n1] = size(V1);
    [m2,n2] = size(V2);
    z1 = trace(V1'*L_1*V1);
    %z2 = trace(V2'*L2*V2);
    z3 = trace(Y1'*(kron_col(V1)-V2));
    z4 = trace(Y2'*(V1'*V1-eye(n1)));
    %z5 = trace(Y3'*(V2'*V2-eye(m2)));
    z6 = mu/2* (norm(kron_col(V1)-V2,'fro')*norm(kron_col(V1)-V2,'fro') ...
        +norm(V1'*V1-eye(n1),'fro')*norm(V1'*V1-eye(n1),'fro') );
    %Z = -z1+z3+z4+z6;
    Z = -z1;
end
