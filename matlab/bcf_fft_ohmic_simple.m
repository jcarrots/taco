function [C, t, meta] = bcf_fft_ohmic_simple(beta, dt, T, omegac, pad_factor, use_pow2)
% Bath correlation C(t) for Ohmic J(ω)=ω e^{-ω/ωc} at temperature 1/beta.
% Grid: t = 0:dt:T (no nmax, no Simpson).
% pad_factor enlarges the FFT length to refine the frequency grid (default 4).
% use_pow2 = true pads to next power-of-two for speed.

    if nargin < 5 || isempty(pad_factor), pad_factor = 4; end
    if nargin < 6 || isempty(use_pow2),   use_pow2   = true; end

    % target time grid
    Nt_time = floor(T/dt) + 1;
    Nfft = max(2, pad_factor*(Nt_time-1));    % at least 2, pad in time
    if use_pow2, Nfft = 2^nextpow2(Nfft); end
    if mod(Nfft,2) ~= 0, Nfft = Nfft + 1; end   % even length

    domega = 2*pi / (Nfft*dt);
    wpos   = (0:(Nfft/2))' * domega;           % length Nfft/2+1

    % positive-frequency half of S(ω) using KMS/detailed-balance form
    % For ω>0: S(ω) = (π/2) * J(ω) / (1 - e^{-βω}) with J(ω)=ω e^{-ω/ωc}
    Spos = zeros(numel(wpos),1);
    Spos(1) = pi/(2*beta);                               % limit ω→0
    if numel(wpos) > 2
        w = wpos(2:end-1);
        Spos(2:end-1) = (pi/2) * (w .* exp(-w/omegac)) ./ (1 - exp(-beta*w));
    end
    wNyq = wpos(end);
    Spos(end) = 0.25*pi*wNyq*exp(-wNyq/omegac)*coth(beta*wNyq/2);  % Nyquist

    % full spectrum S(ω_k), k=0..Nfft-1, with negative side via KMS
    S = zeros(Nfft,1);
    S(1:(Nfft/2+1)) = Spos;
    if Nfft/2 > 1
        KMS  = exp(-beta * wpos(2:end-1));                 % e^{-βω}
        S(Nfft:-1:(Nfft/2+2)) = Spos(2:end-1) .* KMS;      % negative ω
    end

    % time correlation by FFT: C[n] ≈ (dω/π) ∑_k e^{-i ω_k t_n} S_k
    Cfull = fft(S) * (domega/pi);
    %Cfull = ifft(S) * (Nfft * domega / (2*pi));
    % return requested window [0,T]
    C = Cfull(1:Nt_time);
    t = (0:Nt_time-1).' * dt;

    % metadata
    meta.Nfft = Nfft;
    meta.domega = domega;
    meta.wpos = wpos;
    meta.S = S;
    meta.pad_factor = pad_factor;
end
