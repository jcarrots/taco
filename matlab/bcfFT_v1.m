function [bcf,bend]=bcfFT_v1(nmax,ns,beta,dt,omegac,tup)
%%%calculates the BCF for MarkovLP_v9 at temperature 1/beta, using FFT
 %the simplest Ohmic SD with exp cutoff
%tup=2e7/omegac;  %the relative convergence of GW will be 5x10^-7 at ns=20 and tup=8e7/omegac
t_f=dt*ns*nmax*2;   %final time at which we estimate the Markov limit
tf=2*t_f;   %this expansion is done because we will discard the BCF at t<0
            %we get (-t_f,t_f] for FFT; which is folded to [0,2*t_f);
tic
nup=round(tup/tf);
nmaxup=16*nup*nmax*ns;
tf=2*nup*t_f;   %this final time is larger than 2*t_f, because we calculate the BCF
                %on much longer time scale than 2*t_f. 
domega=2*pi/tf;
omegaM=domega*(nmaxup/2);

bcf=zeros(nmaxup,1);
omeg=domega*(1:nmaxup/2-1)';
KSM=exp(-beta*omeg);
bcf(1)=pi/(2*beta);
bcf(2:nmaxup/2)=(pi/2)*omeg.*exp(-omeg/omegac)./(1-KSM);
bcf(end/2+1)=0.25*pi*omegaM*exp(-omegaM/omegac)*coth(beta*omegaM/2);
clear omeg
toc
bcf(end:-1:end/2+2)=bcf(2:nmaxup/2).*KSM;
clear KSM


bcf=fft(bcf)*(domega/pi);
bend=bcf(nmaxup/2+1);
bcf=bcf(1:nmaxup/2);  %center the spectrum
% % T=(dt/4)*(0:nmaxup/2)';  %this are time points but are not
% % needed 
toc
end
