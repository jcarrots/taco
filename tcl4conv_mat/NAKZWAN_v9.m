function GW=NAKZWAN_v9(M,Ia,K,X,Bbath)
%Bbath are coupling operators
nb=numel(Bbath);
[ndim,~,~,~]=size(M);
T=zeros(ndim,ndim,ndim,ndim);

for n=1:ndim
for i=1:ndim
for m=1:ndim
for j=1:ndim

    res=0;
    for iA=1:nb
    A=Bbath{iA};
        for iB=1:nb
        B=Bbath{iB};
            for a=1:ndim
                for b=1:ndim
                    res=res-(B(n,a)*A(a,b)*B(b,i)*A(j,m)*M(b,i,n,a)-A(n,a)*B(a,b)*B(b,i)*A(j,m)*Ia(a,n,i,b)...
                        +B(n,a)*B(a,b)*A(b,i)*A(j,m)*K(i,b,n,a)+A(n,a)*B(a,i)*B(j,b)*A(b,m)*X(a,n,j,b,n,i)...
                        -B(n,a)*A(a,i)*B(j,b)*A(b,m)*X(i,a,j,b,n,i)-A(n,a)*A(a,b)*B(b,i)*B(j,m)*X(b,a,j,m,a,i)...
                        +A(n,a)*B(a,b)*A(b,i)*B(j,m)*X(i,b,j,m,a,i));
                        if j==m
                            for c=1:ndim
                            res=res+(A(n,a)*B(a,b)*A(b,c)*B(c,i)*M(c,i,a,b)+...
                            A(n,a)*B(a,b)*B(b,c)*A(c,i)*K(i,c,a,b)-A(n,a)*A(a,b)*B(b,c)*B(c,i)*Ia(b,a,i,c));
                            end
                        end
                end
            end
        end
    end
    T(n,i,m,j)=res;
end
end
end
end
%GW=T
T=reshape(T,ndim^2,ndim^2);
GW=T+T';
end
% Xones=reshape(eye(ndim),ndim^2,1);
% Iones=find(Xones==1);
% [ITP,JTP,GRP]=find(GW);
% N1=mod(ITP-1,ndim)+1;
% I1=(ITP-N1)/ndim+1;  %(ni) pair
% M1=mod(JTP-1,ndim)+1;
% J1=(JTP-M1)/ndim+1; %(mj) pair 
% iTP=(M1-1)*ndim+N1;  %(nm) pair
% jTP=(J1-1)*ndim+I1;  %(ij) pair
% DW=full(sparse(iTP,jTP,GRP,ndim^2,ndim^2));
% Dsum=sum(DW(Iones,:));
% norm(Dsum)


% D=kron(A,conj(AF))+kron(AF,conj(A));
% %Kossakowski matrix
% [I,J,D]=find(D);
% G=sparse(ndim*floor((I-1)/ndim)+floor((J-1)/ndim)+1,ndim*mod((I-1),ndim)+mod(J-1,ndim)+1,D);
% G=full(G);
