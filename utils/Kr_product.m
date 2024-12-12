function C = Kr_product(A,B)
    nr_A = size(A,1);
    nr_B = size(B,1);
    mul = ones(nr_B,1);
    AA = kron(A,mul);
    BB = repmat(B,nr_A,1);
    C = AA.*BB;
end