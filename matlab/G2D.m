function DW=G2D(GW)
%Permutation from Tensor G(ni,mj) to D(nm,ij)
%--------------------------------------------------------------------------
%Method: to find the index of non-zero element, and build a new tensor by permuting 
%index pair
%--------------------------------------------------------------------------
%A better way to do this is to use the built-in function reshape() and
%permute(), which is 10 times quicker.
%For a random tensor of size (100,100,100,100),the time for G2D is 11.577202
%seconds, the built-in function combination only takes 1.092504 seconds.
%This is because the mod() function is very slow.
%For a sparse matrix, the speed can be comparable. For a diagonal matirx with size (100,100,100,100),
%the time for G2D is 0.147398 while for permute method is 0.077259 seconds.
%For a zero matrix (100,100,100,100), the time is close. G2D is 0.081748;
%time for permute method is 0.078496 seconds.
%--------------------------------------------------------------------------
tic 
[ndim2,~]=size(GW);
ndim=sqrt(ndim2);
[ITP,JTP,GRP]=find(GW); %row and colomn index for non-zero element
N1=mod(ITP-1,ndim)+1;
I1=(ITP-N1)/ndim+1;  %(ni) pair
M1=mod(JTP-1,ndim)+1;
J1=(JTP-M1)/ndim+1; %(mj) pair 
iTP=(M1-1)*ndim+N1;  %(nm) pair
jTP=(J1-1)*ndim+I1;  %(ij) pair
DW=full(sparse(iTP,jTP,GRP,ndim^2,ndim^2));
toc

% tic
% a=reshape(GW,ndim,ndim,ndim,ndim);
% b=permute(a,[1,3,2,4]);
% DW=reshape(b,ndim2,ndim2);
% toc
end

