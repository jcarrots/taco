function [D,Dt,DM]=getAsymptoticALL(alpha,omegac,A,E,Jr,Sr,DW)
%This is a function to calculate Louivillian of Redfield, TCL4, GAME
%Input:
%alpha,omegac,A,E: coupling constant, cutoff frequency, coupling operator, Eigenenergy
%Jr, Sr: Real and Imaginary part of asymptotic spectral density
%DW:dynamical tensor
%output: 
%D,Dt,DM-Louivillian of Redfield, tcl4, GAME
ndim=numel(E);
E=reshape(E,ndim,1);
Xones=eye(ndim);
% Diags=Xones(:)*Xones(:)';
Iones=find(Xones==1);
% Izeros=find(Xones==0);
%alpha=1
AF=alpha*A.*(Jr+1i*Sr).';%filtered operator LAMBDA
Hls=A*AF/2i;
Hls=Hls+Hls';

%REDFIELD
G2=AF(:)*A(:)';G2=G2+G2';  %Relaxation Matrix Eq. 21
D=kron(conj(AF),A)+kron(conj(A),AF); % D_{nm,ij}=G_{ni,mj} Dynamical matrix
G1=Xones(:)*sum(D(Iones,:))/2; %losses in Eq. 21
G2=G2-G1-G1'; %Form Rel Matrix towards the superoperator, will 
%be the realignment
Hu=diag(E)+Hls;
Unitary=-1i*Hu(:)*Xones(:)';
Unitary=Unitary+Unitary'; 
G=G2+Unitary;

% % HLS=(1i/(2*ndim))*reshape(sum(G(:,Iones),2),ndim,ndim);
% % HLS=HLS+HLS';
D=G2D(G);
Dt=D+alpha^2*DW;

%GAME
M=A.*sqrt(2*Jr.');
GM2=M(:)*M(:)';
DM=kron(conj(M),M);
GM1=Xones(:)*sum(DM(Iones,:))/2; %losses in Eq. 21
GM2=GM2-GM1-GM1'; %Form Rel Matrix towards the superoperator, will 
%be the realignment

GM=alpha*GM2+Unitary;
DM=G2D(GM);

%[VW,W]=eigs(D,ndim^2);W=diag(W);
% den=reshape(VW(:,ndim^2),ndim,ndim);den=(den+den')/2;den=den/trace(den)

end


