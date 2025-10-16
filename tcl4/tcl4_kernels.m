function [F, C, R] = tcl4_kernels(G1, G2, Omega, dt, op2)
% TCL_KERNELS  Compute F(t), C(t), R(t) via prefix integrals + Volterra convolution.
%
% [F,C,R] = tcl_kernels(G1, G2, Omega, dt, op2)
%
% Inputs
%   G1    : Γ_{ω1}(t), size N×1  (scalar)  or N×n×m (matrix at each time)
%   G2    : Γ_{ω2}(t), same time length N; if matrix, dims must align with G1
%   Omega : scalar  (Ω = ω1+ω2+ω3)
%   dt    : time step (uniform grid)
%   op2   : operation applied to Γ_{ω2} in F(t). One of:
%           'T' (transpose, default), 'I' (identity), 'conj' (complex conj), 'H' (Hermitian)
%
% Outputs (same shape as a timewise product Γ1 @ Γ2)
%   F(t) = ∫_0^t [Γ1(t) - Γ1(t-s)] Γ2^{op2}(s) e^{-iΩ(t-s)} ds
%   C(t) = ∫_0^t [Γ1(t) - Γ1(t-s)] Γ2^*(s)    e^{-iΩ(t-s)} ds
%   R(t) = ∫_0^t [Γ1(t) - Γ1(t-s)] Γ2(t-s)    e^{-iΩ(t-s)} ds
%
% Notes
% - Time is the FIRST dimension.
% - Scalars can be given as N×1; they are promoted internally to N×1×1.
% - Requires R2016b+ (implicit expansion). For speed, R2020b+ uses pagemtimes.
% - Volterra convolution is linear (FFT) and returns the causal part.

    if nargin < 5, op2 = 'T'; end

    % Ensure 3-D arrays (N×n×m); scalars become N×1×1
    G1 = ensure3d(G1);
    G2 = ensure3d(G2);

    N = size(G1,1);
    t = (0:N-1).' * dt;

    % Phases (broadcast over trailing dims)
    phase_minus = reshape(exp(-1i*Omega*t), [N 1 1]);   % e^{-iΩ t}
    phase_plus  = reshape(exp(+1i*Omega*t), [N 1 1]);   % e^{+iΩ t}

    % ---------- F(t) ----------
    B_F   = apply_op_g2(G2, op2);                       % Γ2^{op2}
    A_F   = prefix_int_left(B_F .* phase_plus, dt);     % ∫ Γ2^{op2}(s) e^{+iΩ s} ds
    term1 = time_matmul(G1 .* phase_minus, A_F);        % Γ1(t)e^{-iΩ t} times prefix integral
    term2 = volterra_conv_matmul(G1 .* phase_minus, B_F, dt);  % (Γ1 e^{-iΩ·}) ⋆ Γ2^{op2}
    F     = term1 - term2;

    % ---------- C(t) ----------
    B_C   = conj(G2);                                   % Γ2^*
    A_C   = prefix_int_left(B_C .* phase_plus, dt);     % ∫ Γ2^*(s) e^{+iΩ s} ds
    term1c = time_matmul(G1 .* phase_minus, A_C);
    term2c = volterra_conv_matmul(G1 .* phase_minus, B_C, dt);
    C      = term1c - term2c;

    % ---------- R(t) ----------
    A_R   = prefix_int_left(G2 .* phase_minus, dt);     % ∫ Γ2(τ) e^{-iΩ τ} dτ
    term1r = time_matmul(G1, A_R);                      % Γ1(t) * that prefix integral
    P      = time_matmul(G1, G2);                       % Γ1(τ) @ Γ2(τ)
    term2r = prefix_int_left(P .* phase_minus, dt);     % ∫ Γ1(τ)Γ2(τ) e^{-iΩ τ} dτ
    R      = term1r - term2r;
end

% ==================== helpers ====================

function out = prefix_int_left(y, dt)
    % Left-Riemann cumulative integral along time (dim 1)
    out = dt * cumsum(y, 1);
end

function X = ensure3d(X)
    % Promote to N×n×m with time on dim 1; scalars -> N×1×1
    if ndims(X) < 3
        X = reshape(X, [size(X,1) 1 1]);
    end
end

function B = apply_op_g2(G2, op)
    % Apply 'I' | 'T' | 'conj' | 'H' to Γ2 (time on dim 1)
    switch lower(op)
        case 'i'
            B = G2;
        case 't'
            B = permute(G2, [1 3 2]);          % transpose trailing dims
        case 'conj'
            B = conj(G2);
        case 'h'                               % Hermitian
            B = permute(conj(G2), [1 3 2]);
        otherwise
            error('Unknown op2: %s (use ''I'',''T'',''conj'',''H'')', op);
    end
end

function Y = time_matmul(A, B)
    % For each time t: Y(t) = A(t) @ B(t).
    % Shapes: A (N,n,m), B (N,m,p) -> Y (N,n,p).
    A = ensure3d(A); B = ensure3d(B);
    Ap = permute(A, [2 3 1]);  % -> n×m×N (pages over time)
    Bp = permute(B, [2 3 1]);  % -> m×p×N
    Yp = page_gemm(Ap, Bp);    % n×p×N
    Y  = permute(Yp, [3 1 2]); % -> N×n×p
end

function Y = volterra_conv_matmul(F, G, dt)
    % Volterra convolution with matrix multiplication in the inner dim:
    %   Y(t) = ∫_0^t F(t-s) @ G(s) ds
    % Shapes: F (N,n,m), G (N,m,p) -> Y (N,n,p)
    N = size(F,1);
    L = 2*N - 1;
    Ff = fft(F, L, 1);               % FFT along time (dim 1)
    Gf = fft(G, L, 1);

    % Multiply at each frequency with contraction over the inner dimension
    Ffp = permute(Ff, [2 3 1]);      % n×m×L
    Gfp = permute(Gf, [2 3 1]);      % m×p×L
    Hfp = page_gemm(Ffp, Gfp);       % n×p×L

    Hf  = permute(Hfp, [3 1 2]);     % L×n×p
    y_full = ifft(Hf, [], 1);
    Y = dt * y_full(1:N, :, :);      % causal (Volterra) part
end

function C = page_gemm(A, B)
    % Pagewise matrix multiply A(:,:,k)*B(:,:,k) for all k.
    % Uses pagemtimes if available; falls back to a loop otherwise.
    try
        C = pagemtimes(A, B);                % R2020b+
    catch
        K = size(A,3);
        C = zeros(size(A,1), size(B,2), K, 'like', A);
        for k = 1:K
            C(:,:,k) = A(:,:,k) * B(:,:,k);
        end
    end
end
