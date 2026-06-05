# Port of Alpha_comp_alpha_slowing.f90 NBI_flag=2 (fusion alpha + 1 MeV NBI pencil beams).

"""NBI pencil-beam injection parameters (Fortran `NBI_model=2`)."""
Base.@kwdef struct NBIBeamParams{T<:Real}
    Pow_NBI::T = T(30.0)              # total beam power [MW]
    E_nbi::T = T(1.0)                 # beam energy [MeV]
    lambda_NBI::Union{T,Nothing} = nothing  # attenuation length [m]; default 2.5*rmin(end)
    Rmaj_NBI::Union{T,Nothing} = nothing    # tangent major radius [m]; default Rmaj at edge
    i_NBI_max::Int = 7
    NBI_model::Int = 2                 # 2: 2*i_NBI_max-1 equal beamlets
    M_DT::T = T(2.5)                  # effective mass factor for tau_s (Fortran M_DT/4 / 1/2)^2
end

"""
    nbi_pencil_beam_source(input; nbi, Rmaj) -> S_nbi

Volume-averaged NBI source S02 [10^19 m^-3 s^-1] from multi-beamlet pencil paths
(Fortran `Alpha_comp_alpha_slowing`, `NBI_flag=2`).
"""
function nbi_pencil_beam_source(
    input::AlphaInput{T};
    nbi::NBIBeamParams{T}=NBIBeamParams{T}(),
    Rmaj::AbstractVector{T}=input.Rmaj,
) where {T<:Real}
    rho, rmin, ne = input.rho, input.rmin, input.ne
    n = length(rho)
    Vp = _vprime(input.volume, rmin)
    a = rmin[end]
    Rmaj0 = Rmaj[end]
    Rmaj_NBI = nbi.Rmaj_NBI === nothing ? Rmaj0 : nbi.Rmaj_NBI
    lambda = nbi.lambda_NBI === nothing ? T(2.5) * a : nbi.lambda_NBI
    Icur_NBI = nbi.Pow_NBI / nbi.E_nbi / T(1.6022e-19)
    ne_edge = ne[n]

    S02 = zeros(T, n)
    for i_NBI in 1:nbi.i_NBI_max
        h_NBI = a * rho[i_NBI]
        z01, z02, z03, z04 = _nbi_path_lengths(Rmaj, Rmaj_NBI, h_NBI, rho, a, n)
        I_path = _nbi_beam_current(Icur_NBI, i_NBI, nbi)
        lnI = log(max(I_path, eps(T)))
        # path 1: inward from edge above beam index
        for i in n-1:-1:(i_NBI + 1)
            Rmaj[i] > Rmaj_NBI || continue
            dz = abs(z01[i] - z01[i-1])
            lnI -= dz / lambda * (ne[i] / ne_edge)
            drho = max(abs(rho[i] - rho[i-1]) * a, eps(T))
            S02[i] += exp(lnI) * dz / lambda * (ne[i] / ne_edge) / drho / max(Vp[i], eps(T))
        end
        lnI = log(max(I_path, eps(T)))
        # path 2
        for i in (i_NBI + 1):n
            Rmaj[i] > Rmaj_NBI || continue
            dz = abs(z02[i] - z02[i-1])
            lnI -= dz / lambda * (ne[i] / ne_edge)
            drho = max(abs(rho[i] - rho[i-1]) * a, eps(T))
            S02[i] += exp(lnI) * dz / lambda * (ne[i] / ne_edge) / drho / max(Vp[i], eps(T))
        end
        lnI = log(max(I_path, eps(T)))
        # path 3
        for i in n-1:-1:(i_NBI + 1)
            Rmaj[i] > Rmaj_NBI || continue
            dz = abs(z03[i] - z03[i-1])
            lnI -= dz / lambda * (ne[i] / ne_edge)
            drho = max(abs(rho[i] - rho[i-1]) * a, eps(T))
            S02[i] += exp(lnI) * dz / lambda * (ne[i] / ne_edge) / drho / max(Vp[i], eps(T))
        end
        lnI = log(max(I_path, eps(T)))
        # path 4
        for i in (i_NBI + 1):n
            Rmaj[i] > Rmaj_NBI || continue
            dz = abs(z04[i] - z04[i-1])
            lnI -= dz / lambda * (ne[i] / ne_edge)
            drho = max(abs(rho[i] - rho[i-1]) * a, eps(T))
            S02[i] += exp(lnI) * dz / lambda * (ne[i] / ne_edge) / drho / max(Vp[i], eps(T))
        end
    end
    S02[1] = S02[2]
    S02[n] = S02[n] > 0 ? S02[n] : S02[n-1]
    return S02
end

function _nbi_beam_current(Icur_NBI::T, i_NBI::Int, nbi::NBIBeamParams{T}) where {T<:Real}
    if nbi.NBI_model == 2
        I = Icur_NBI / T(2 * nbi.i_NBI_max - 1)
        return i_NBI > 1 ? 2 * I : I
    elseif nbi.NBI_model == 1
        return Icur_NBI / T(nbi.i_NBI_max)
    else
        return Icur_NBI
    end
end

function _nbi_path_lengths(Rmaj, Rmaj_NBI, h_NBI, rho, a, n)
    T = eltype(Rmaj)
    z01 = zeros(T, n)
    z02 = zeros(T, n)
    z03 = zeros(T, n)
    z04 = zeros(T, n)
    Rn = Rmaj[n]
    for i in 1:n
        Ri = Rmaj[i]
        z_edge = sqrt(max((Rn - h_NBI + a * rho[n])^2 - Rmaj_NBI^2, zero(T)))
        z01[i] = z_edge - sqrt(max((Ri - h_NBI + a * rho[i])^2 - Rmaj_NBI^2, zero(T)))
        z02[i] = z_edge - sqrt(max((Ri + h_NBI - a * rho[i])^2 - Rmaj_NBI^2, zero(T)))
        z03[i] = z_edge + sqrt(max((Ri + h_NBI - a * rho[i])^2 - Rmaj_NBI^2, zero(T)))
        z04[i] = z_edge + sqrt(max((Ri - h_NBI + a * rho[i])^2 - Rmaj_NBI^2, zero(T)))
    end
    return z01, z02, z03, z04
end

"""
    slowing_down_nbi(ne, Te; E_nbi, S_nbi, Z1_nbi, ln_lambda, M_DT) -> (n_cl, T_equiv, E_c_hat)

NBI slowing-down classical density from prescribed source `S_nbi` [10^19 m^-3 s^-1]
(Fortran alpha2 branch with mass/charge corrections).
"""
function slowing_down_nbi(
    ne::AbstractVector{T}, Te::AbstractVector{T}, S_nbi::AbstractVector{T};
    E_nbi::T, Z1_nbi::T, ln_lambda::T, M_DT::T=T(2.5),
) where {T<:Real}
    n = length(ne)
    n_cl = zeros(T, n)
    T_equiv = zeros(T, n)
    E_c_hat = zeros(T, n)
    c_ec = (4 * 1836)^(one(T) / 3) * (3 * sqrt(T(pi)) * Z1_nbi / 4)^(T(2) / 3) * (M_DT / 4)^(one(T) / 3)
    mass_fac = (M_DT / 4) / (one(T) / 2)^2
    for i in 1:n
        E_c_hat[i] = (Te[i] / E_nbi) * T(1e-3) * c_ec
        a = sqrt(max(E_c_hat[i], eps(T)))
        I2 = (one(T) / 3) * log((1 + a^3) / a^3)
        I4 = one(T) / 2 - a^2 * ((one(T) / 6) * log((1 - a + a^2) / (1 + a)^2) +
                                 (1 / sqrt(T(3))) * (atan((2 - a) / (a * sqrt(T(3)))) + T(pi) / 6))
        T_equiv[i] = (T(2) / 3) * I4 / I2 * E_nbi * T(1e3)
        tau_ee = T(1.088e-3) * Te[i]^1.5 / ne[i] / ln_lambda
        tau_s = T(1836) * tau_ee * mass_fac
        n_cl[i] = tau_s * I2 * S_nbi[i]
    end
    return n_cl, T_equiv, E_c_hat
end

"""Z1 correction for NBI species in DT plasma (Fortran `NBI_flag=2`)."""
function nbi_Z1(M_DT::T=T(2.5)) where {T<:Real}
    return T(5 // 3) * (M_DT / 4) / (M_DT / 2.5)
end
