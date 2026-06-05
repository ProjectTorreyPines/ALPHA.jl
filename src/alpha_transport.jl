# Port of GACODE Alpha_transport.f90 stiff critical-gradient-model (CGM) relaxation.

"""Parameters for [`stiff_cgm_transport`](@ref) (defaults follow Alpha_transport.f90)."""
Base.@kwdef struct AlphaTransportParams{T<:Real}
    delta0::T = 0.01
    delta1::T = 0.0
    rdelta0::T = 0.5
    D_bkg::T = 0.001
    D_TAE::T = 7.4
    SDsink::T = 1.0
    relax::T = 5.0e-4
    relax_f::T = 0.005
    n_iter::Int = 2000
    tol::T = 1.0e-3
    l_crit_smooth::Bool = true
    use_angioni_bkg::Bool = false
    angioni_pinch_fac::T = 1.0
    angioni_negative::Bool = true
    Q_fus::T = 10.0
    i_tot_TAE::Int = -1          # -1: pressure; 0: density; 1: joint total pressure (dual EP)
    adapt_D_TAE::Bool = false    # Fortran adapt_D_TAE_flag
    use_ql_diffusivity::Bool = false
    ql_params::Union{QLDiffusivityParams{T},Nothing} = nothing
    he_ash_params::Union{HeAshParams{T},Nothing} = nothing
end

"""Result of the stiff-CGM transport solve."""
Base.@kwdef struct StiffCGMResult{T<:Real}
    n_tran::Vector{T}
    p_tran::Vector{T}
    flux::Vector{T}
    D_alpha::Vector{T}
    rg_n_tran::Vector{T}
    rg_p_tran::Vector{T}
    rg_n_th::Vector{T}
    rg_p_th::Vector{T}
    error::T
    n_iter::Int
    n_tran2::Vector{T} = T[]      # NBI / second EP species
    D_alpha2::Vector{T} = T[]
    D_ql::Vector{T} = T[]
    he_ash::Union{HeAshResult{T},Nothing} = nothing
    ql_state::Union{QLDiffusivityState{T},Nothing} = nothing
end

const _KEV19_TO_KPA = 0.16022  # p [10^19·keV] = n*T; p [10 kPa] = n*T*0.16022

"""Fortran-style radial gradient: ``-(f[i+1]-f[i-1])/(a*Δρ)`` on interior points."""
function _radial_grad(f::AbstractVector{T}, rho::AbstractVector{T}, a::T) where {T<:Real}
    n = length(f)
    g = zeros(T, n)
    for i in 2:n-1
        g[i] = -(f[i+1] - f[i-1]) / (a * (rho[i+1] - rho[i-1]))
    end
    g[n] = -(f[n] - f[n-1]) / (a * (rho[n] - rho[n-1]))
    return g
end

"""``V' ≈ dV/dr`` [m²] from enclosed volume."""
function _vprime(volume::AbstractVector{T}, rmin::AbstractVector{T}) where {T<:Real}
    return _dArea(volume, rmin)
end

"""Smooth jagged critical pressure gradients (l_crit_smooth in Fortran)."""
function _smooth_crit_pressure!(rg_p_th::AbstractVector{T}, dr::AbstractVector{T}) where {T<:Real}
    n = length(rg_p_th)
    n < 3 && return rg_p_th
    d2f = zeros(T, n)
    d2b = zeros(T, n)
    d2f[2:n-1] = (rg_p_th[3:n] .- rg_p_th[2:n-1]) ./ dr[2:n-1]
    d2b[2:n-1] = (rg_p_th[2:n-1] .- rg_p_th[1:n-2]) ./ dr[1:n-2]
    for i in 2:n-1
        denom = min(abs(d2f[i]), abs(d2b[i])) + T(1e-4)
        if abs(d2f[i] - d2b[i]) / denom > T(0.5)
            rg_p_th[i] = T(0.4) * rg_p_th[i] + T(0.3) * (rg_p_th[i-1] + rg_p_th[i+1])
        end
    end
    return rg_p_th
end

"""Build threshold gradients from TGLF-EP critical inputs (i_threshold=10)."""
function _threshold_gradients(
    n_cl::AbstractVector{T}, T_equiv::AbstractVector{T},
    dndr_crit::Union{AbstractVector{T},Nothing}, dpdr_crit::Union{AbstractVector{T},Nothing},
    rho::AbstractVector{T}, rmin::AbstractVector{T}, a::T;
    critgrad_method::Symbol, l_crit_smooth::Bool,
) where {T<:Real}
    n = length(rho)
    rg_n_th = zeros(T, n)
    rg_p_th = zeros(T, n)
    dr = _dr_grid(rmin)

    if critgrad_method === :density
        dndr_crit === nothing && error("critgrad_method=:density requires dndr_crit")
        rg_n_th .= dndr_crit
        for i in 2:n-1
            rg_p_th[i] = T_equiv[i] * rg_n_th[i] * _KEV19_TO_KPA * (
                1 + (T_equiv[i+1] - T_equiv[i-1]) / T_equiv[i] /
                    (n_cl[i+1] - n_cl[i-1]) * n_cl[i])
        end
        rg_p_th[1] = rg_p_th[2]
        rg_p_th[n] = rg_p_th[n-1]
    elseif critgrad_method === :pressure
        dpdr_crit === nothing && error("critgrad_method=:pressure requires dpdr_crit")
        # dpdr_crit from TJLFEP is 10 kPa/m; convert to 10^19·keV/m for internal p gradients
        rg_p_th .= dpdr_crit ./ _KEV19_TO_KPA
        dndr_crit === nothing || (rg_n_th .= dndr_crit)
    else
        error("unknown critgrad_method=$critgrad_method")
    end

    l_crit_smooth && _smooth_crit_pressure!(rg_p_th, dr)
    return rg_n_th, rg_p_th
end

function _dr_grid(rmin::AbstractVector{T}) where {T<:Real}
    n = length(rmin)
    dr = zeros(T, n)
    for i in 1:n-1
        dr[i] = rmin[i+1] - rmin[i]
    end
    dr[n] = dr[n-1]
    return dr
end

"""Angioni background diffusivity from alpha energy flux (normed to chi_eff)."""
function _angioni_D_bkg(
    flux_source::AbstractVector{T}, Vp::AbstractVector{T},
    ne::AbstractVector{T}, Te::AbstractVector{T}, Ti::AbstractVector{T},
    ni::AbstractVector{T}, dr::AbstractVector{T}, E_alpha::T, Q_fus::T,
) where {T<:Real}
    n = length(flux_source)
    chi = zeros(T, n)
    for i in 2:n-1
        denom = T(0.5) * ni[i] * (-Ti[i+1] + Ti[i-1]) / (2 * dr[i]) +
                T(0.5) * ne[i] * (-Te[i+1] + Te[i-1]) / (2 * dr[i])
        abs(denom) < eps(T) && continue
        chi[i] = (1 + 5 / Q_fus) * flux_source[i] * (E_alpha * 1000) / denom
    end
    chi[1] = chi[2]
    chi[n] = chi[n-1]
    D = zeros(T, n)
    for i in 1:n
        x = Te[i] / (E_alpha * 1000)
        D[i] = chi[i] * (T(0.02) + 4.5x + 8.0x^2 + 350.0x^3)
    end
    return D
end

"""Pinch correction coefficient C_p_alpha (interior points)."""
function _C_p_alpha(Rmaj::AbstractVector{T}, Te::AbstractVector{T}, E_c_hat::AbstractVector{T},
                    dr::AbstractVector{T}; angioni_exp::T=1.0) where {T<:Real}
    n = length(Te)
    C = zeros(T, n)
    for i in 2:n-1
        dT = (-Te[i+1] + Te[i-1]) / (2 * dr[i])
        ec = max(E_c_hat[i], eps(T))
        C[i] = (3 / 2) * Rmaj[i] * dT / Te[i] *
               (1 / (1 + 1 / ec^(angioni_exp * 1.5)) / log(1 + 1 / ec^1.5) - 1)
    end
    C[n] = (3 / 2) * Rmaj[n] * (-Te[n] + Te[n-1]) / dr[n] / Te[n] *
           (1 / (1 + 1 / max(E_c_hat[n], eps(T))^(angioni_exp * 1.5)) /
                 log(1 + 1 / max(E_c_hat[n], eps(T))^1.5) - 1)
    return C
end

"""
    stiff_cgm_transport(input, n_cl, T_equiv, S0, crit_grad; kwargs...) -> StiffCGMResult

Stiff CGM relaxation port of `Alpha_transport.f90` (inward integration from the edge).

`crit_grad` supplies `dndr_crit` and/or `dpdr_crit` (TJLFEP outputs). Use
`critgrad_method=:density` or `:pressure` to select the threshold branch.
"""
function stiff_cgm_transport(
    input::AlphaInput{T},
    n_cl::AbstractVector{T},
    T_equiv::AbstractVector{T},
    S0::AbstractVector{T},
    crit_grad;
    params::AlphaTransportParams{T}=AlphaTransportParams{T}(),
    critgrad_method::Symbol=:density,
    dndr_crit=_getgrad(crit_grad, :dndr_crit),
    dpdr_crit=_getgrad(crit_grad, :dpdr_crit),
    n_cl2::Union{AbstractVector{T},Nothing}=nothing,
    T_equiv2::Union{AbstractVector{T},Nothing}=nothing,
    S02::Union{AbstractVector{T},Nothing}=nothing,
    crit_grad2=nothing,
    ql_modes::Vector{QLModeInput{T}}=QLModeInput{T}[],
    ql_state::Union{QLDiffusivityState{T},Nothing}=nothing,
) where {T<:Real}
    dual = n_cl2 !== nothing && T_equiv2 !== nothing && S02 !== nothing
    dual || (n_cl2 = nothing; T_equiv2 = nothing; S02 = nothing)
    rho = input.rho
    rmin = input.rmin
    n = length(rho)
    a = rmin[end]
    dr = _dr_grid(rmin)
    Vp = _vprime(input.volume, rmin)

    dndr = dndr_crit === nothing ? nothing : collect(T, dndr_crit)
    dpdr = dpdr_crit === nothing ? nothing : collect(T, dpdr_crit)

    rg_n_th, rg_p_th = _threshold_gradients(
        n_cl, T_equiv, dndr, dpdr, rho, rmin, a;
        critgrad_method, l_crit_smooth=params.l_crit_smooth)

    # Initial guess (delta0 profile + edge BC)
    n_tran = similar(n_cl)
    for i in 1:n
        n_tran[i] = n_cl[i] * (1 + params.delta0 * (rho[i] - params.rdelta0) / params.rdelta0)
    end
    n_tran[n] = params.delta1 * n_cl[n]

    p_cl = n_cl .* T_equiv .* _KEV19_TO_KPA
    p_cl2 = dual ? n_cl2 .* T_equiv2 .* _KEV19_TO_KPA : nothing
    D_alpha = fill(params.D_bkg, n)
    D_alpha2 = dual ? fill(params.D_bkg, n) : T[]
    n_tran2 = dual ? similar(n_cl2) : T[]
    if dual
        for i in 1:n
            n_tran2[i] = n_cl2[i] * (1 + params.delta0 * (rho[i] - params.rdelta0) / params.rdelta0)
        end
        n_tran2[n] = params.delta1 * n_cl2[n]
    end
    flux = zeros(T, n)
    flux2 = dual ? zeros(T, n) : nothing
    error = T(Inf)
    n_done = 0
    D_TAE_cur = params.D_TAE

    ql_p = params.ql_params === nothing ? QLDiffusivityParams{T}() : params.ql_params
    ql_state === nothing && params.use_ql_diffusivity && (ql_state = QLDiffusivityState{T}(n, dual ? 2 : 1, ql_p.km_max))
    QLdiff = params.use_ql_diffusivity ? zeros(T, dual ? 2 : 1, n) : nothing
    dndr2 = dual ? _getgrad(crit_grad2 === nothing ? crit_grad : crit_grad2, :dndr_crit) : nothing
    dpdr2 = dual ? _getgrad(crit_grad2 === nothing ? crit_grad : crit_grad2, :dpdr_crit) : nothing
    rg_n_th2 = rg_p_th2 = nothing
    if dual
        dndr2v = dndr2 === nothing ? dndr : collect(T, dndr2)
        dpdr2v = dpdr2 === nothing ? dpdr : collect(T, dpdr2)
        rg_n_th2, rg_p_th2 = _threshold_gradients(
            n_cl2, T_equiv2, dndr2v, dpdr2v, rho, rmin, a;
            critgrad_method, l_crit_smooth=params.l_crit_smooth)
    end

    # Optional Angioni background D
    D_bkg_ang = if params.use_angioni_bkg
        flux_src = _source_flux(S0, Vp, dr, n_tran, n_cl, params.SDsink)
        _angioni_D_bkg(flux_src, Vp, input.ne, input.Te, input.Ti, input.ni, dr, input.E_alpha, params.Q_fus)
    else
        fill(params.D_bkg, n)
    end

    Rmaj = isempty(input.Rmaj) ? _default_Rmaj(rmin, n) : input.Rmaj
    C_p = params.use_angioni_bkg ? _C_p_alpha(Rmaj, input.Te, fill(T(0.1), n), dr) : zeros(T, n)
    dndr_vec = dndr === nothing ? copy(rg_n_th) : dndr

    for iter in 1:params.n_iter
        n_prev = copy(n_tran)
        D_prev = copy(D_alpha)

        rg_n_tran = _radial_grad(n_tran, rho, a)
        p_tran = n_tran .* T_equiv .* _KEV19_TO_KPA
        rg_p_tran = _radial_grad(p_tran, rho, a)

        rg_p_tot = nothing
        if dual && params.i_tot_TAE == 1
            p_tot = n_tran .* T_equiv .* _KEV19_TO_KPA .+ n_tran2 .* T_equiv2 .* _KEV19_TO_KPA
            rg_p_tot = _radial_grad(p_tot, rho, a)
            rg_p_tot_th = sqrt.(rg_p_th .* rg_p_th2)
        end

        if dual
            rg_n_tran2 = _radial_grad(n_tran2, rho, a)
            p_tran2 = n_tran2 .* T_equiv2 .* _KEV19_TO_KPA
            rg_p_tran2 = _radial_grad(p_tran2, rho, a)
        end

        # D_alpha with optional Angioni pinch, stiff TAE, optional QL
        for i in 1:n
            D_alpha[i] = params.D_bkg
            dual && (D_alpha2[i] = params.D_bkg)
            if params.use_angioni_bkg && i != 1 && i != n
                pinch = params.angioni_pinch_fac * C_p[i] / Rmaj[i] * n_tran[i] / max(rg_n_tran[i], eps(T))
                params.angioni_negative && (pinch = max(pinch, T(-0.9)))
                D_alpha[i] = D_bkg_ang[i] * (1 + pinch)
            elseif params.use_angioni_bkg
                D_alpha[i] = D_bkg_ang[i]
            end
            if params.i_tot_TAE == 0
                excess = max(rg_n_tran[i] - rg_n_th[i], zero(T))
                D_alpha[i] += D_TAE_cur * excess * a / max(n_cl[i], eps(T))
            elseif params.i_tot_TAE == -1
                excess = max(rg_p_tran[i] - rg_p_th[i], zero(T))
                D_alpha[i] += D_TAE_cur * excess * a / max(p_cl[i], eps(T))
            elseif params.i_tot_TAE == 1 && dual && rg_p_tot !== nothing
                p_tot_cl = p_cl[i] + p_cl2[i]
                excess = max(rg_p_tot[i] - rg_p_tot_th[i], zero(T))
                D_alpha[i] += D_TAE_cur * excess * a / max(p_tot_cl, eps(T))
            end
            if dual
                if params.i_tot_TAE == 0
                    ex2 = max(rg_n_tran2[i] - rg_n_th2[i], zero(T))
                    D_alpha2[i] += D_TAE_cur * ex2 * a / max(n_cl2[i], eps(T))
                elseif params.i_tot_TAE == -1
                    ex2 = max(rg_p_tran2[i] - rg_p_th2[i], zero(T))
                    D_alpha2[i] += D_TAE_cur * ex2 * a / max(p_cl2[i], eps(T))
                elseif params.i_tot_TAE == 1
                    D_alpha2[i] = D_alpha[i]
                end
            end
        end
        if params.use_angioni_bkg
            D_alpha[1] = D_alpha[2]
            D_alpha[n] = D_alpha[n-1]
            if dual
                D_alpha2[1] = D_alpha2[2]
                D_alpha2[n] = D_alpha2[n-1]
            end
        end
        D_alpha .= params.relax_f .* D_alpha .+ (1 - params.relax_f) .* D_prev

        if params.use_ql_diffusivity
            den = dual ? hcat(n_tran, n_tran2)' : reshape(n_tran, 1, n)
            dndr2v = dual ? (dndr2 === nothing ? (rg_n_th2 === nothing ? dndr_vec : copy(rg_n_th2)) : collect(T, dndr2)) : dndr_vec
            sp1 = (; n_cl=n_cl, T_equiv=T_equiv, dndr_crit=dndr_vec, modes=ql_modes)
            species = dual ? [sp1, (; n_cl=n_cl2, T_equiv=T_equiv2, dndr_crit=dndr2v)] : [sp1]
            ql_diffusivity!(QLdiff, den, ql_state, ql_p, input, species;
                Rmaj=Rmaj, phase=iter == 1 ? :init : :update, modes_species1=ql_modes)
            D_alpha .+= QLdiff[1, :]
            dual && (D_alpha2 .+= QLdiff[2, :])
        end

        # Particle flux (with slowing-down sink)
        net_sink = 1 .- params.SDsink .* n_tran ./ max.(n_cl, eps(T))
        flux[1] = zero(T)
        for i in 2:n
            flux[i] = Vp[i-1] / max(Vp[i], eps(T)) * flux[i-1] +
                      T(0.5) * dr[i] / max(Vp[i], eps(T)) *
                      (Vp[i] * S0[i] * net_sink[i] + Vp[i-1] * S0[i-1] * net_sink[i-1])
        end
        flux0 = copy(flux)

        # Implicit inward integration: n_i = n_{i+1} + dr*(Γ_i+Γ_{i+1})/(D_i+D_{i+1})
        for i in n-1:-1:1
            Dsum = max(D_alpha[i] + D_alpha[i+1], eps(T))
            n_tran[i] = n_tran[i+1] + dr[i] * (flux[i] + flux[i+1]) / Dsum
            n_tran[i] = max(n_tran[i], zero(T))
        end

        # Convergence metric
        err = zero(T)
        for i in 2:n-1
            e = (n_tran[i] - n_prev[i]) / max(abs(n_prev[i]), eps(T))
            err += e^2
        end
        error = sqrt(err / max(n - 2, 1))
        n_done = iter

        n_tran .= params.relax .* n_tran .+ (1 - params.relax) .* n_prev

        if dual
            n_prev2 = copy(n_tran2)
            net2 = 1 .- params.SDsink .* n_tran2 ./ max.(n_cl2, eps(T))
            flux2[1] = zero(T)
            for i in 2:n
                flux2[i] = Vp[i-1] / max(Vp[i], eps(T)) * flux2[i-1] +
                           T(0.5) * dr[i] / max(Vp[i], eps(T)) *
                           (Vp[i] * S02[i] * net2[i] + Vp[i-1] * S02[i-1] * net2[i-1])
            end
            for i in n-1:-1:1
                Dsum = max(D_alpha2[i] + D_alpha2[i+1], eps(T))
                n_tran2[i] = n_tran2[i+1] + dr[i] * (flux2[i] + flux2[i+1]) / Dsum
                n_tran2[i] = max(n_tran2[i], zero(T))
            end
            n_tran2 .= params.relax .* n_tran2 .+ (1 - params.relax) .* n_prev2
        end

        if params.adapt_D_TAE && error > eps(T)
            D_TAE_cur *= max(T(0.999), 1 + log(T(1e-3) / error))
        end

        error < params.tol && break
    end

    p_tran = n_tran .* T_equiv .* _KEV19_TO_KPA
    rg_n_tran = _radial_grad(n_tran, rho, a)
    rg_p_tran = _radial_grad(p_tran, rho, a)

    he_res = nothing
    he_p = params.he_ash_params === nothing ? HeAshParams{T}(; enabled=false) : params.he_ash_params
    if he_p.enabled
        he_res = he_ash_transport(input, n_tran, n_cl, S0; params=he_p, Rmaj=Rmaj)
    end

    D_ql_out = params.use_ql_diffusivity && QLdiff !== nothing ? collect(QLdiff[1, :]) : T[]

    return StiffCGMResult{T}(;
        n_tran, p_tran, flux, D_alpha,
        rg_n_tran, rg_p_tran, rg_n_th, rg_p_th, error, n_iter=n_done,
        n_tran2=dual ? n_tran2 : T[], D_alpha2=dual ? D_alpha2 : T[],
        D_ql=D_ql_out, he_ash=he_res, ql_state)
end

function _source_flux(S0, Vp, dr, n_tran, n_cl, SDsink)
    T = eltype(S0)
    n = length(S0)
    flux = zeros(T, n)
    flux[1] = zero(T)
    for i in 2:n
        flux[i] = Vp[i-1] / max(Vp[i], eps(T)) * flux[i-1] +
                  T(0.5) * dr[i] / max(Vp[i], eps(T)) *
                  (Vp[i] * S0[i] + Vp[i-1] * S0[i-1])
    end
    return flux
end

"""Default major radius [m] if not stored in AlphaInput (2 m typical)."""
_default_Rmaj(rmin::AbstractVector{T}, n::Int) where {T<:Real} = fill(T(2.0), n)
