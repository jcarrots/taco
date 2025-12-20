% Hamiltonian, eigen-decomposition, coupling operator
H = [0 1; 1 0]/2;
[V, EigMat] = eig(H);
Eig = diag(EigMat);
A = V' * [1 0; 0 -1] * V / 2;

omegac = 10.0;
beta   = 0.5;

nmax = 2;
dt   = 0.0025/4;

% choose the largest ns you care about and build BCF once
ns_list = 2:2:2^14;                % times of interest
ns_max  = max(ns_list);
tup     = 2^10;
tic
[bcf_raw_long, bend_long] = bcfFT_v1(nmax, ns_max, beta, dt, omegac, tup);

% build full time-series kernels once up to T_max = 2*nmax*ns_max*dt
[F_all, C_all, R_all, Gamma_all, map] = ...
    compute_FCR_timeseries(Eig, dt, nmax, ns_max, bcf_raw_long, bend_long);

% containers for outputs at selected times
nCases = numel(ns_list);
M_i  = cell(nCases,1);
I_i  = cell(nCases,1);
K_i  = cell(nCases,1);
X_i  = cell(nCases,1);
Gt_i = cell(nCases,1);
t_tcl = zeros(nCases,1);

for ii = 1:nCases
    ns = ns_list(ii);
    tidx = 2*nmax*ns + 1;         % index in time series
    t_tcl(ii) = map.t(tidx);

    % map Γ_ω(tidx) to N×N
    G_end_u = Gamma_all(tidx, :).';                 % nf×1
    G_end   = reshape(G_end_u(map.ij), [map.N, map.N]);
    Gt_i{ii} = G_end;

    % assemble A2/A3/A4 at this time using the ij mapping
    F_t = squeeze(F_all(tidx, :, :, :));            % nf×nf×nf
    C_t = squeeze(C_all(tidx, :, :, :));
    R_t = squeeze(R_all(tidx, :, :, :));

    A2_1 = reshape(F_t(map.ij, map.ij, map.ij), [map.N map.N map.N map.N map.N map.N]);
    A3_1 = reshape(R_t(map.ij, map.ij, map.ij), [map.N map.N map.N map.N map.N map.N]);
    A4_1 = reshape(C_t(map.ij, map.ij, map.ij), [map.N map.N map.N map.N map.N map.N]);

    Fd_t = {nmax, A2_1, A3_1, A4_1};

    % build M,I,K,X at this time
    [M, I, K, X] = MIKX(map.nf, map.omegas, map.N, Fd_t);
    M_i{ii} = M;  I_i{ii} = I;  K_i{ii} = K;  X_i{ii} = X;
end

% example downstream usage at each selected time
GW_i  = zeros(4,4,nCases);
RedT  = zeros(4,4,nCases);
TCL4T = zeros(4,4,nCases);
for ii = 1:nCases
    GW_i(:,:,ii) = NAKZWAN_v9(M_i{ii}, I_i{ii}, K_i{ii}, X_i{ii}, {A});
    [Dr1, Dt1, DM] = getAsymptoticALL( ...
        0.5, omegac, A, Eig, real(Gt_i{ii}), imag(Gt_i{ii}), G2D(GW_i(:,:,ii)) );
    RedT(:,:,ii)  = Dr1;
    TCL4T(:,:,ii) = Dt1;
end
toc
