function VV1 = kron_col(V1)
    [m,n] = size(V1);
    VV1 = zeros(m*m,n);
    for i = 1:n
        v1i = V1(:,i);
        VV1(:,i) = kron(v1i,v1i);
    end
end



