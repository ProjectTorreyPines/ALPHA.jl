# Port of GACODE Alpha_QLdiffusivity.f90 — quasi-linear CGM EP diffusivity from TGLF-EP
# mode growth rates and QL diffusivity weights (diff_star), with E_hat/Z_hat relaxation.

"""Inputs for one QL AE mode (from TGLF-EP or a test model)."""
Base.@kwdef struct QLModeInput{T<:Real}
    gamma_star::Union{Vector{T},T} = T(0.1)   # growth rate scale [1/s] in gB units (Fortran default)
    diff_star::Union{Vector{T},T} = T(1.0)    # QL diffusivity weight [m^2/s]
    rg_n_crit::Union{Vector{T},Nothing} = nothing  # critical -dn/dr; default: copy species threshold
    crit_index_shift::Int = 0                 # radial index shift for island-chain modes (Fortran i_del2)
    crit_scale::T = T(1.0)                    # frac_crit multiplier
end

"""Runtime state for [`ql_diffusivity!`](@ref) (Fortran `Alpha_use_sav_QLdiffEP`)."""
mutable struct QLDiffusivityState{T<:Real}
    initialized::Bool
    n_species::Int
    km_max::Int
    E_hat::Matrix{T}              # km_max × n
    Z_hat::Matrix{T}
    time_called::T
    sav_rg_n_th::Vector{Vector{T}} # per species
    sav_rg_n_sd::Vector{Vector{T}}
    sav_Ln_sd::Vector{Vector{T}}
    sav_LT_sd::Vector{Vector{T}}
    sav_gamma_star::Array{T,3}    # is × km × n
    sav_diff_star::Array{T,3}
    sav_rg_n_crit::Array{T,3}
end

function QLDiffusivityState{T}(n::Int, n_species::Int=1, km_max::Int=5) where {T<:Real}
    QLDiffusivityState{T}(
        false, n_species, km_max,
        zeros(T, km_max, n), zeros(T, km_max, n), zero(T),
        [zeros(T, n) for _ in 1:n_species],
        [zeros(T, n) for _ in 1:n_species],
        [zeros(T, n) for _ in 1:n_species],
        [zeros(T, n) for _ in 1:n_species],
        zeros(T, n_species, km_max, n),
        zeros(T, n_species, km_max, n),
        zeros(T, n_species, km_max, n),
    )
end

"""Parameters for quasi-linear diffusivity ([`ql_diffusivity!`](@ref))."""
Base.@kwdef struct QLDiffusivityParams{T<:Real}
    km_max::Int = 5
    D_bkg::T = T(0.001)
    C_nl::T = zero(T)
    CZ_nl::T = zero(T)
    C_R::T = T(2.0)
    use_f_cor::Bool = false
    omega_RBF1::T = T(1.0)
    F_mode_drive::Vector{T} = T[1.0, 1.0, 1.0, 1.0, 1.0]
    dt_update::T = T(1.0e-3)
    E_hat_init::T = T(0.01)
    dual_species_coupling::Bool = true   # NBI_flag=2 style F_frac split
    island_delays::Vector{Int} = [0, 20, 5, 10, 15]  # per-mode index shifts (modes 2–5)
    island_frac_crit::Vector{T} = [T(1), T(1), T(1), T(1), T(1)]
end

function _fill_mode_field!(out::AbstractVector{T}, val::Union{AbstractVector{T},T}, n::Int) where {T<:Real}
    if val isa AbstractVector
        length(val) == n || error("QL mode field length $(length(val)) != grid $n")
        out .= val
    else
        fill!(out, val)
    end
    return out
end

"""Build per-mode critical gradients (Fortran island-chain defaults when modes not supplied)."""
function _build_ql_crit_star!(
    sav_rg_n_crit::Array{T,3},
    is::Int,
    rg_n_th::AbstractVector{T},
    modes::Vector{QLModeInput{T}},
    km_max::Int,
) where {T<:Real}
    n = length(rg_n_th)
    for km in 1:km_max
        m = km <= length(modes) ? modes[km] : QLModeInput{T}()
        base = m.rg_n_crit === nothing ? copy(rg_n_th) : collect(T, m.rg_n_crit)
        scale = m.crit_scale
        shift = m.crit_index_shift
        for i in 1:n
            j = max(1, i - shift)
            sav_rg_n_crit[is, km, i] = scale * base[j]
        end
    end
    return sav_rg_n_crit
end

function _Ln_from_rg(n::T, rg::T) where {T<:Real}
    abs(rg) < eps(T) && return T(Inf)
    return n / rg
end

"""
    ql_diffusivity!(QLdiffEP, den, state, params, input, species; kwargs...) -> QLDiffusivityState

Compute quasi-linear EP diffusivity [m^2/s] for one or two species.

`den` is `n_species × n` transported densities [10^19 m^-3]. `species` is a vector of named
tuples or structs with `(n_cl, T_equiv, dndr_crit, modes)` per species.

Call with `phase=:init` once, then `phase=:update` each stiff-CGM iteration (or time step).
"""
function ql_diffusivity!(
    QLdiffEP::AbstractMatrix{T},
    den::AbstractMatrix{T},
    state::QLDiffusivityState{T},
    params::QLDiffusivityParams{T},
    input::AlphaInput{T},
    species::AbstractVector;
    Rmaj::AbstractVector{T}=input.Rmaj,
    phase::Symbol=:update,
    modes_species1::Vector{QLModeInput{T}}=QLModeInput{T}[],
) where {T<:Real}
    rho, rmin = input.rho, input.rmin
    n = length(rho)
    a = rmin[end]
    n_species = size(den, 1)
    km_max = params.km_max
    size(QLdiffEP) == (n_species, n) || error("QLdiffEP size must match den")

    if phase === :init || !state.initialized
        _ql_init!(state, params, input, species, Rmaj, modes_species1)
        state.initialized = true
    end

    if phase === :init
        QLdiffEP .= params.D_bkg
        return state
    end

    # transported gradients + f_cor
    rg_tran = [zeros(T, n) for _ in 1:n_species]
    Ln_tran = [zeros(T, n) for _ in 1:n_species]
    f_cor = ones(T, n_species, n)
    for is in 1:n_species
        rg_tran[is] = _radial_grad(vec(den[is, :]), rho, a)
        for i in 2:n
            Ln_tran[is][i] = _Ln_from_rg(den[is, i], rg_tran[is][i])
        end
        Ln_tran[is][1] = Ln_tran[is][2]
        if params.use_f_cor && is <= length(state.sav_Ln_sd)
            for i in 1:n
                num = 1 + Ln_tran[is][i] / state.sav_LT_sd[is][i] - params.C_R * Ln_tran[is][i] / Rmaj[i]
                denf = 1 + state.sav_Ln_sd[is][i] / state.sav_LT_sd[is][i] -
                       params.C_R * state.sav_Ln_sd[is][i] / Rmaj[i]
                f_cor[is, i] = num / max(abs(denf), eps(T))
            end
        end
    end

    gamma_hat = zeros(T, n_species, km_max, n)
    RBF1 = ones(T, n_species, km_max, n)
    F_drive = params.F_mode_drive
    length(F_drive) < km_max && (F_drive = [F_drive; fill(F_drive[end], km_max - length(F_drive))])

    F_frac_one = n_species == 1 ? one(T) : (params.dual_species_coupling ? T(0.5) : one(T))

    for km in 1:km_max
        Fd = F_drive[km]
        for i in 1:n
            gamma_AE = zero(T)
            gamma_Z = zero(T)
            for is in 1:n_species
                F_frac = is == 1 ? F_frac_one : (one(T) - F_frac_one)
                excess = Fd * rg_tran[is][i] - F_frac * state.sav_rg_n_crit[is, km, i]
                g = state.sav_gamma_star[is, km, i] * f_cor[is, i] * excess / max(state.sav_rg_n_sd[is][i], eps(T))
                gamma_hat[is, km, i] = g
                gamma_AE += g
                gamma_Z += state.sav_gamma_star[is, km, i] * f_cor[is, i] *
                           F_frac * state.sav_rg_n_crit[is, km, i] / max(state.sav_rg_n_sd[is][i], eps(T))
            end
            RBF1[1, km, i] = gamma_hat[1, km, i] / (gamma_hat[1, km, i]^2 + params.omega_RBF1^2)
            n_species == 2 && (RBF1[2, km, i] = gamma_hat[2, km, i] / (gamma_hat[2, km, i]^2 + params.omega_RBF1^2))
            # advance E_hat, Z_hat (Fortran i_sav=0 block)
            state.E_hat[km, i] += params.dt_update * 2 * gamma_AE * state.E_hat[km, i] -
                                  params.dt_update * 2 * params.C_nl * state.E_hat[km, i]^2 -
                                  params.dt_update * 2 * params.CZ_nl * state.E_hat[km, i] * state.Z_hat[km, i]
            state.E_hat[km, i] = max(state.E_hat[km, i], zero(T))
            if params.CZ_nl > 0
                Zprev = state.Z_hat[km, i]
                state.Z_hat[km, i] += params.dt_update * 2 * params.CZ_nl * state.E_hat[km, i] * Zprev -
                                      params.dt_update * 2 * gamma_Z * Zprev
                state.Z_hat[km, i] = max(state.Z_hat[km, i], zero(T))
            end
        end
    end
    state.time_called += params.dt_update

    QLdiffEP .= zero(T)
    for km in 1:km_max
        for is in 1:n_species
            for i in 2:n-1
                QLdiffEP[is, i] += state.E_hat[km, i] * state.sav_diff_star[is, km, i] *
                                   f_cor[is, i] * RBF1[is, km, i]
            end
        end
    end
    for is in 1:n_species
        QLdiffEP[is, 1] = QLdiffEP[is, 2]
        QLdiffEP[is, n] = QLdiffEP[is, n-1]
    end
    QLdiffEP ./= T(km_max)
    QLdiffEP .+= params.D_bkg
    return state
end

function _ql_init!(
    state::QLDiffusivityState{T},
    params::QLDiffusivityParams{T},
    input::AlphaInput{T},
    species::AbstractVector,
    Rmaj::AbstractVector{T},
    modes_species1::Vector{QLModeInput{T}},
) where {T<:Real}
    rho, rmin = input.rho, input.rmin
    n = length(rho)
    a = rmin[end]
    n_species = length(species)
    km_max = params.km_max
    state.n_species = n_species
    state.km_max = km_max
    state.E_hat .= params.E_hat_init
    state.Z_hat .= params.E_hat_init
    state.time_called = zero(T)

    default_modes = QLModeInput{T}[]
    for km in 1:km_max
        shift = km == 1 ? 0 : (km <= length(params.island_delays) ? params.island_delays[km] : 0)
        frac = km <= length(params.island_frac_crit) ? params.island_frac_crit[km] : one(T)
        push!(default_modes, QLModeInput{T}(; crit_index_shift=shift, crit_scale=frac))
    end
    modes1 = isempty(modes_species1) ? default_modes : modes_species1

    for (is, sp) in enumerate(species)
        n_cl = sp.n_cl
        T_eq = sp.T_equiv
        dndr = sp.dndr_crit
        rg_sd = _radial_grad(n_cl, rho, a)
        rg_T = _radial_grad(T_eq, rho, a)
        state.sav_rg_n_sd[is] .= rg_sd
        state.sav_rg_n_th[is] .= dndr
        for i in 2:n
            state.sav_Ln_sd[is][i] = _Ln_from_rg(n_cl[i], rg_sd[i])
            state.sav_LT_sd[is][i] = T_eq[i] / max(rg_T[i], eps(T))
        end
        state.sav_Ln_sd[is][1] = state.sav_Ln_sd[is][2]
        state.sav_LT_sd[is][1] = state.sav_LT_sd[is][2]

        modes = is == 1 ? modes1 : [
            QLModeInput{T}(; gamma_star=T(0.1), diff_star=T(1.0), crit_scale=one(T)) for _ in 1:km_max]
        if sp isa NamedTuple && haskey(sp, :modes) && sp.modes !== nothing
            modes = sp.modes
        end

        for km in 1:km_max
            m = modes[km]
            _fill_mode_field!(view(state.sav_gamma_star, is, km, :), m.gamma_star, n)
            _fill_mode_field!(view(state.sav_diff_star, is, km, :), m.diff_star, n)
        end
        _build_ql_crit_star!(state.sav_rg_n_crit, is, state.sav_rg_n_th[is], modes, km_max)
        if is == 2
            for km in 1:km_max
                state.sav_rg_n_crit[2, km, :] .= state.sav_rg_n_th[2]
            end
        end
    end
    return state
end

"""Build QL mode table from uniform TGLF-EP critical gradient (test / default)."""
function default_ql_modes(dndr_crit::AbstractVector{T}; km_max::Int=5,
                          params::QLDiffusivityParams{T}=QLDiffusivityParams{T}()) where {T<:Real}
    modes = QLModeInput{T}[]
    for km in 1:km_max
        shift = km == 1 ? 0 : params.island_delays[min(km, length(params.island_delays))]
        frac = params.island_frac_crit[min(km, length(params.island_frac_crit))]
        push!(modes, QLModeInput{T}(;
            rg_n_crit=collect(T, dndr_crit), crit_index_shift=shift, crit_scale=frac))
    end
    return modes
end
