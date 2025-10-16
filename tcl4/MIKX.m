function [M_jkpq,I_jkpq,K_jkpq,X_ikpqrs]=MIKX(nf,omegas,N,Fd)
% JPhysA-116840.R5 E(12) to E(15)
% copy tensor, aka order 3 identity tensor, aka order 3 kronecker delta
% Will be needed later
I3 = zeros([N,N,N]);
I3((0:N-1).*(N^2 + N + 1) +ones(1,N)) = 1;

Iome = zeros([nf,nf,nf]);
Iome((0:nf-1).*(nf^2 + nf + 1) +ones(1,nf)) = 1;

I2 = eye(N);


A_jkpqrs=Fd{2};
B_jkpqrs=Fd{3};
C_jkpqrs=Fd{4};

%(G_jq(t)-G...)G_{jq}(t_1).'e^{-omega_{jk}(t-t_1)}
% M1=permute(tensorprod( ...
%     tensorprod(A_jkpqrs,I3,[4,6],[1,2]), ...
%     I3,[1,5],[1,2]), ...
%     [4,2,3,1]);
M1=permute(tensorprod( ...
    tensorprod(A_jkpqrs,I3,[3,6],[1,2]), ...
    I3,[1,5],[1,2]), ...
    [4,2,3,1]);
%(G_jq(t)-G_jq(t-t_1))G_pq(t-t_1)e^{-omega_{jk}(t-t_1)}
M2=permute(tensorprod( ...
    tensorprod(B_jkpqrs,I3,[2,5],[1,2]), ...
    I3,[3,5],[1,2]), ...
    [1,3,2,4]);
%(G_jq(t)-G...)G_{jq}(t_1).'e^{-omega_{pq}(t-t_1)}


M_jkpq=M1-M2;
%(G_jq(t)-G_jq(t-t_1))G_pq(t_1)e^{-omega_{jp}(t-t_1)}
I1=permute(tensorprod( ...
    tensorprod(A_jkpqrs,I3,[3,6],[1,2]), ...
    I3,[2,4],[1,2]), ...
    [1,4,2,3]);

I_jkpq=I1;
%(G_jq(t)-G_jq(t-t_1))G_pq(t-t_1)e^{omega_{kp}(t-t_1)}
K1=permute(tensorprod( ...
    tensorprod(B_jkpqrs,I3,[1,6],[1,2]), ...
    I3,[3,4],[1,2]), ...
    [3,1,2,4]);
K_jkpq=K1;
X_ikpqrs=C_jkpqrs+B_jkpqrs;
%Final Operators
end