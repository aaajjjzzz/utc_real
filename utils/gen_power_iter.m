function [W, obj_rec, iter] = gen_power_iter(A,B)
%GEN_POWER_ITER  to solve QPSM to get the loval minimum 
%Input:
%       L   -   Symmetric matrix m*m
%       B   -   matrix m*k
%Output:
%       W   -   the optical solution
% v1.0 written by FeiQi in 20210815
   
% para set and initial
    [m, k] = size(B);
    max_iter = 1e3;
    iter = 1;
    tol = 1e-5;
    L = randn(m,m);
    L = L'*L;
    W = L(:,k);
    [~,eigs] = eig(A);
    alpha = max(diag(eigs));
    A_hat = alpha*eye(m)-A;
% starting the loop till convergence
    while true
        W_t = W;
        M = 2*A_hat*W + 2*B;
        [U,S,V] = svd(M);
        W = U(:,1:k)*V';
        if max(max(abs(W_t-W)))<tol || iter > max_iter
            break;
        end
        obj_rec(iter) = trace(W'* A * W) + trace(2*W'*B);
        iter = iter + 1;
    end
    fprintf('¹²µü´ú%d´În', iter);

end

