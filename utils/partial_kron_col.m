function PV = partial_kron_col(V1,V2,Y1,mu)
    [m1,n1] = size(V1);
    %[m2,n2] = size(V2);
    
    PV = zeros(m1,n1);
    for i = 1:n1
        V1_i = V1(:,i);
        V2_i = V2(:,i);
        Y1_i = Y1(:,i);
        PV(:,i) = ( kron(eye(m1),V1_i) + kron(V1_i,eye(m1)) )' * (kron_col(V1_i)-V2_i+Y1_i/mu);
    end
    
end