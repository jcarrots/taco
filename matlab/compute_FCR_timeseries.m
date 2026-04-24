function [F_all, C_all, R_all, Gamma_all, map] = compute_FCR_timeseries(Eig, dt, nmax, ns, bcf_raw, bend)
% Build full time-series F(t), C(t), R(t) and Γ_ω(t)
% Requires tcl_kernels.m on path

    % eigenvalues and frequency set
    if isvector(Eig), E = Eig(:); else, E = diag(Eig); end
    N = numel(E);
    omegaij = E - E.';
    omegas   = reshape(omegaij, [N^2, 1]);
    [omegas_u, ~, ij] = unique(omegas, 'stable');
    nf = numel(omegas_u);

    % time grid
    Nt = 2*nmax*ns + 1;
    t  = (0:Nt-1).' * dt;

    % unpack BCF to a single 1D correlation on [0, T]
%     nr  = 8*nmax;
%     nup = numel(bcf_raw) / (nr*ns);
%     if abs(nup - round(nup)) > eps
%         error('bcf_raw length inconsistent with (nr, ns).');
%     end
%     bcf = reshape(bcf_raw, nr, []);
%     bcf(nr+1,1:ns*nup-1) = bcf(1,2:ns*nup);
%     bcf(nr+1,ns*nup)     = bend;
    C_all = bcf_raw(1:Nt);
    if numel(C_all) < Nt
        error('BCF is too short: need at least %d points', Nt);
    end
    C = C_all(1:Nt);

    % Γ_ω(t) for all unique ω
    Phase_plus = exp(1i * (t * omegas_u.'));     % Nt × nf
    Gamma_all  = dt * cumsum( C .* Phase_plus, 1 );  % Nt × nf

    % full time-series F, C, R over all frequency triples
    F_all = complex(zeros(Nt, nf, nf, nf));
    C_allker = complex(zeros(Nt, nf, nf, nf));
    R_all = complex(zeros(Nt, nf, nf, nf));

    for a = 1:nf
        G1 = Gamma_all(:, a);
        for b = 1:nf
            G2 = Gamma_all(:, b);
            for c = 1:nf
                Omega = omegas_u(a) + omegas_u(b) + omegas_u(c);
                [Fv, Cv, Rv] = tcl4_kernels(G1, G2, Omega, dt, 'I');  % returns full vectors
                F_all(:, a, b, c)   = Fv;
                C_allker(:, a, b, c)= Cv;
                R_all(:, a, b, c)   = Rv;
            end
        end
    end

    % pack outputs
    C_all = C_allker;  % name matching
    map.t         = t;
    map.dt        = dt;
    map.N         = N;
    map.nf        = nf;
    map.omegas    = omegas;      % N^2 × 1 (pairwise diffs)
    map.omegas_u  = omegas_u;    % nf × 1 (unique set)
    map.ij        = ij;          % N^2 mapping into omegas_u
end
