"""
    ALPHA

Fast-particle (energetic-particle, EP) transport solver for TGLF-EP.

This is a Julia port of the steady-state GACODE `Alpha` model. It consumes the
TGLF-EP critical-gradient profiles (from `TJLFEP.runTHD`) together with the
background plasma in an IMAS `dd`, and integrates the critical gradients into the
energetic-particle radial profiles: density `n_EP`, pressure `p_EP`, temperature
`T_EP = p_EP / n_EP`, plus the associated EP particle/energy flux.

Physics ported from `\$CFS/m3739/gacode_add_d3d/Alpha`:
  * `Alpha_comp_alpha_slowing.f90` -> [`slowing_down`](@ref): classical
    slowing-down density, equivalent-Maxwellian temperature `T_alpha_equiv`, and
    cross-over energy `E_c_hat`.
  * `Alpha_transport.f90` (stiff-CGM relaxation) -> [`stiff_cgm_transport`](@ref):
    iterative flux-matching with stiff AE diffusivity above the TGLF-EP critical
    gradient threshold.
  * [`integrate_crit_grad`](@ref): analytic marginal-profile limit (fast; used when
    `solver=:marginal`).

Also ported: quasi-linear EP diffusivity ([`ql_diffusivity!`](@ref)), fusion+NBI dual EP
([`nbi_pencil_beam_source`](@ref), [`slowing_down_nbi`](@ref)), and He ash transport
([`he_ash_transport`](@ref)).
"""
module ALPHA

using LinearAlgebra
using Printf
import IMAS
import GACODE

export run_alpha, AlphaResult, AlphaInput
export stiff_cgm_transport, StiffCGMResult, AlphaTransportParams
export ql_diffusivity!, QLDiffusivityState, QLDiffusivityParams, QLModeInput, default_ql_modes
export nbi_pencil_beam_source, slowing_down_nbi, NBIBeamParams, nbi_Z1
export he_ash_transport, HeAshParams, HeAshResult
export integrate_crit_grad, slowing_down, load_DT_sigma_v
export read_crit_grad, load_crit_grad

const _DATA_DIR = normpath(joinpath(@__DIR__, "data"))

# ──────────────────────────────────────────────────────────────────────────────
# DT fusion reactivity table (ported asset DT_sigma_v.dat: Ti [keV], <sigma v> [m^3/s])
# ──────────────────────────────────────────────────────────────────────────────
"""
    load_DT_sigma_v() -> (Tgrid, sigma_v)

Load the deuterium-tritium fusion reactivity `<sigma v>` [m^3/s] versus ion
temperature [keV] from the packaged `data/DT_sigma_v.dat` (the Fortran
`DT_sigma_v.dat` asset).
"""
function load_DT_sigma_v()
    path = joinpath(_DATA_DIR, "DT_sigma_v.dat")
    Tgrid = Float64[]
    sigv = Float64[]
    for (k, line) in enumerate(eachline(path))
        k == 1 && continue                       # header: "DT temp [keV]  <sigma*v> [m^3/s]"
        parts = split(strip(line))
        length(parts) < 2 && continue
        push!(Tgrid, parse(Float64, parts[1]))
        push!(sigv, parse(Float64, parts[2]))
    end
    return Tgrid, sigv
end

# ──────────────────────────────────────────────────────────────────────────────
# small utilities
# ──────────────────────────────────────────────────────────────────────────────
"""Piecewise-linear interpolation of `(x, y)` evaluated at `xq` (clamped at ends)."""
function _interp(x::AbstractVector, y::AbstractVector, xq::AbstractVector)
    @assert length(x) == length(y) "interp: x and y length mismatch"
    out = similar(xq, promote_type(eltype(y), eltype(xq)))
    n = length(x)
    for (k, q) in enumerate(xq)
        if q <= x[1]
            out[k] = y[1]
        elseif q >= x[n]
            out[k] = y[n]
        else
            j = searchsortedlast(x, q)
            j = clamp(j, 1, n - 1)
            t = (q - x[j]) / (x[j+1] - x[j])
            out[k] = y[j] * (1 - t) + y[j+1] * t
        end
    end
    return out
end

"""Cumulative trapezoidal integral of `y` over `x`, returned as a profile with the same length (out[1]=0)."""
function _cumtrapz(x::AbstractVector, y::AbstractVector)
    n = length(x)
    out = zeros(promote_type(eltype(x), eltype(y)), n)
    for i in 2:n
        out[i] = out[i-1] + 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────────────
# TGLF-EP critical-gradient file readers (alpha_dndr_crit.input / alpha_dpdr_crit.input)
# ──────────────────────────────────────────────────────────────────────────────
"""
    read_crit_grad(path) -> (header, values)

Parse a TGLF-EP critical-gradient profile file, i.e. the `alpha_dndr_crit.input`
/ `alpha_dpdr_crit.input` files written by TJLFEP (`TJLFEP.write_crit_grad`) or
the Fortran `TGLFEP_driver`.

The first line is the descriptive `header` (`Density critical gradient ...` /
`Pressure critical gradient ...`); the remaining lines are the profile
`values`.

Both on-disk layouts are auto-detected:

  * Fortran / current-TJLFEP format — one value per line (`F12.4`). This is the
    format the Fortran `Alpha` solver reads.
  * Legacy TJLFEP format — line 2 is a Julia array literal `[v1, v2, ...]`.
"""
function read_crit_grad(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("read_crit_grad: empty file $path")
    header = strip(lines[1])
    body = length(lines) >= 2 ? strip(lines[2]) : ""
    vals = Float64[]
    if startswith(body, "[")
        # legacy Julia array literal on a single line: [v1, v2, ...]
        for tok in split(strip(body, ['[', ']']), ",")
            s = strip(tok)
            isempty(s) && continue
            push!(vals, parse(Float64, s))
        end
    else
        # Fortran F12.4 layout: one value per line after the header
        for ln in @view lines[2:end]
            s = strip(ln)
            isempty(s) && continue
            x = tryparse(Float64, s)
            x === nothing || push!(vals, x)
        end
    end
    return String(header), vals
end

"""
    load_crit_grad(; dndr=nothing, dpdr=nothing) -> NamedTuple

Read TGLF-EP critical-gradient files into the `crit_grad` argument accepted by
[`run_alpha`](@ref). Pass the path to the density-gradient file (`dndr`,
`alpha_dndr_crit.input`) and/or the pressure-gradient file (`dpdr`,
`alpha_dpdr_crit.input`); a missing keyword yields a `nothing` field.

    crit_grad = load_crit_grad(; dndr="alpha_dndr_crit.input",
                                 dpdr="alpha_dpdr_crit.input")
    run_alpha(dd, rho, crit_grad; method=:density)
"""
function load_crit_grad(; dndr::Union{Nothing,AbstractString}=nothing,
                        dpdr::Union{Nothing,AbstractString}=nothing)
    dndr_crit = dndr === nothing ? nothing : read_crit_grad(dndr)[2]
    dpdr_crit = dpdr === nothing ? nothing : read_crit_grad(dpdr)[2]
    return (; dndr_crit, dpdr_crit)
end

# ──────────────────────────────────────────────────────────────────────────────
# input / output containers
# ──────────────────────────────────────────────────────────────────────────────
"""
    AlphaInput

Background-plasma + control inputs for [`run_alpha`](@ref), assembled on the
ALPHA radial grid `rho` (rho_tor_norm). All densities in 10^19 m^-3, temperatures
in keV, lengths in m.
"""
Base.@kwdef mutable struct AlphaInput{T<:Real}
    rho::Vector{T}            # rho_tor_norm grid
    rmin::Vector{T}           # midplane minor radius [m] vs rho
    ne::Vector{T}             # electron density [10^19 m^-3]
    Te::Vector{T}             # electron temperature [keV]
    Ti::Vector{T}             # (main) ion temperature [keV]
    ni::Vector{T}             # main-ion density [10^19 m^-3] (for fusion source)
    volume::Vector{T}         # enclosed volume [m^3] vs rho (for flux normalization)
    E_alpha::T = T(3.5)       # EP birth energy [MeV] (alpha = 3.5)
    Z1::T = T(5 // 3)         # slowing-down charge factor (5/3 for alphas in 50/50 DT)
    ln_lambda::T = T(17)      # Coulomb logarithm
    Rmaj::Vector{T} = T[]    # major radius [m] on rho grid (filled by AlphaInput(dd) if empty)
end

"""
    AlphaResult

Output of [`run_alpha`](@ref). EP radial profiles on `rho` plus diagnostics.
Densities in 10^19 m^-3, temperatures/pressures in keV / (10^19 m^-3 · keV),
particle flux in 10^19 m^-2 s^-1 and energy flux in keV·10^19 m^-2 s^-1.
"""
Base.@kwdef struct AlphaResult{T<:Real}
    rho::Vector{T}
    n_EP::Vector{T}             # transport-relaxed EP density [10^19 m^-3]
    p_EP::Vector{T}             # EP pressure [10^19 m^-3 · keV]
    T_EP::Vector{T}             # EP temperature [keV]
    flux_particle::Vector{T}    # EP particle flux [10^19 m^-2 s^-1]
    flux_energy::Vector{T}      # EP energy flux [keV · 10^19 m^-2 s^-1]
    # diagnostics
    n_classical::Vector{T}      # source-limited (no-transport) slowing-down density
    T_alpha_equiv::Vector{T}    # equivalent-Maxwellian slowing-down temperature [keV]
    E_c_hat::Vector{T}          # cross-over energy E_c/E_alpha
    S0::Vector{T}               # alpha source [10^19 m^-3 s^-1]
    transport_active::Vector{Bool}  # where AE transport flattens the profile to marginal
    # stiff-CGM diagnostics (when solver=:stiff)
    stiff_error::T
    stiff_n_iter::Int
    D_alpha::Vector{T}
    D_ql::Vector{T}
    n_EP2::Vector{T}           # second EP species (NBI) when ep_mode=:fusion_nbi
    n_He::Vector{T}           # helium ash when He transport enabled
end

_getgrad(crit_grad, key) = crit_grad isa AbstractDict ? get(crit_grad, key, get(crit_grad, String(key), nothing)) :
                           (hasproperty(crit_grad, key) ? getproperty(crit_grad, key) : nothing)

"""Flux-surface area dV/dr [m^2] via central differences of `volume` w.r.t. `rmin`."""
function _dArea(volume::AbstractVector{T}, rmin::AbstractVector{T}) where {T<:Real}
    n = length(volume)
    area = zeros(T, n)
    for i in 1:n
        if i == 1
            area[i] = (volume[2] - volume[1]) / max(rmin[2] - rmin[1], eps(T))
        elseif i == n
            area[i] = (volume[n] - volume[n-1]) / max(rmin[n] - rmin[n-1], eps(T))
        else
            area[i] = (volume[i+1] - volume[i-1]) / max(rmin[i+1] - rmin[i-1], eps(T))
        end
    end
    return area
end

include("alpha_ql_diffusivity.jl")
include("nbi_physics.jl")
include("he_ash_transport.jl")
include("alpha_transport.jl")

# ──────────────────────────────────────────────────────────────────────────────
# slowing-down physics  (port of Alpha_comp_alpha_slowing.f90, NBI_flag=0 path)
# ──────────────────────────────────────────────────────────────────────────────
"""
    slowing_down(ne, Te, Ti, ni; E_alpha, Z1, ln_lambda)
        -> (n_classical, T_alpha_equiv, E_c_hat, S0)

Classical fusion-alpha slowing-down profiles (per radius), faithfully ported from
`Alpha_comp_alpha_slowing.f90`:

  * cross-over energy `E_c_hat = (Te/E_alpha)·1e-3·(4·1836)^(1/3)·(3√π·Z1/4)^(2/3)`
  * slowing-down integrals `I2`, `I4` (with `a = √E_c_hat`)
  * equivalent Maxwellian temperature `T_alpha_equiv = (2/3)(I4/I2)·E_alpha·1e3` [keV]
  * approximate fusion source `S0 = 2.5e-6·ni²·Ti²` [10^19 m^-3 s^-1]
  * slowing-down time `τ_s = 1836·τ_ee`, `τ_ee = 1.088e-3·Te^1.5/ne/lnΛ`
  * classical density `n_classical = S0·τ_s·I2` [10^19 m^-3]
"""
function slowing_down(ne::AbstractVector{T}, Te::AbstractVector{T}, Ti::AbstractVector{T},
                      ni::AbstractVector{T}; E_alpha::T, Z1::T, ln_lambda::T) where {T<:Real}
    n = length(ne)
    E_c_hat = zeros(T, n)
    T_equiv = zeros(T, n)
    S0 = zeros(T, n)
    n_cl = zeros(T, n)
    c_ec = (4 * 1836)^(one(T) / 3) * (3 * sqrt(T(pi)) * Z1 / 4)^(T(2) / 3)
    for i in 1:n
        E_c_hat[i] = (Te[i] / E_alpha) * T(1e-3) * c_ec
        a = sqrt(E_c_hat[i])
        I2 = (one(T) / 3) * log((1 + a^3) / a^3)
        I4 = one(T) / 2 - a^2 * ((one(T) / 6) * log((1 - a + a^2) / (1 + a)^2) +
                                 (1 / sqrt(T(3))) * (atan((2 - a) / (a * sqrt(T(3)))) + T(pi) / 6))
        T_equiv[i] = (T(2) / 3) * I4 / I2 * E_alpha * T(1e3)
        S0[i] = T(2.5e-6) * ni[i]^2 * Ti[i]^2
        tau_ee = T(1.088e-3) * Te[i]^1.5 / ne[i] / ln_lambda
        tau_s = T(1836) * tau_ee
        n_cl[i] = S0[i] * tau_s * I2
    end
    return n_cl, T_equiv, E_c_hat, S0
end

# ──────────────────────────────────────────────────────────────────────────────
# critical-gradient (stiff-CGM) marginal profile integration
# ──────────────────────────────────────────────────────────────────────────────
"""
    integrate_crit_grad(rmin, dXdr_crit; X_edge=0) -> X_marginal

Integrate a critical gradient magnitude `dXdr_crit` (a positive `-dX/dr`) inward
from the edge to obtain the marginal (transport-limited) profile

    X_marginal(r) = X_edge + ∫_r^a dXdr_crit dr' .

This is the steady-state limit that the Fortran stiff-CGM relaxation solver
(`Alpha_transport.f90`) converges to: where the source-limited profile would be
steeper than critical, AE transport flattens it to exactly the marginal profile.
"""
function integrate_crit_grad(rmin::AbstractVector{T}, dXdr_crit::AbstractVector{T};
                             X_edge::T=zero(T)) where {T<:Real}
    n = length(rmin)
    X = zeros(T, n)
    X[n] = X_edge
    for i in (n-1):-1:1
        X[i] = X[i+1] + T(0.5) * (dXdr_crit[i] + dXdr_crit[i+1]) * (rmin[i+1] - rmin[i])
    end
    return X
end

# ──────────────────────────────────────────────────────────────────────────────
# background-plasma extraction from an IMAS dd
# ──────────────────────────────────────────────────────────────────────────────
"""
    AlphaInput(dd, rho; E_alpha=3.5, Z1=5/3, ln_lambda=17) -> AlphaInput

Build an [`AlphaInput`](@ref) from an IMAS `dd`, interpolating the background
plasma (ne, Te, Ti, main-ion density) and geometry (minor radius, enclosed
volume) onto the requested `rho` (rho_tor_norm) grid.
"""
function AlphaInput(dd::IMAS.dd, rho::AbstractVector{T};
                    E_alpha::Real=3.5, Z1::Real=5 // 3, ln_lambda::Real=17) where {T<:Real}
    cp1d = dd.core_profiles.profiles_1d[]
    rho_cp = cp1d.grid.rho_tor_norm
    ne = cp1d.electrons.density_thermal ./ 1e19
    Te = cp1d.electrons.temperature ./ 1e3
    # main (first) thermal ion
    ion = cp1d.ion[1]
    Ti = ion.temperature ./ 1e3
    ni = ion.density_thermal ./ 1e19

    eqt1d = dd.equilibrium.time_slice[].profiles_1d
    rho_eq = eqt1d.rho_tor_norm
    rmin_eq = (eqt1d.r_outboard .- eqt1d.r_inboard) ./ 2
    vol_eq = eqt1d.volume

    rhov = collect(T, rho)
    rminv = _interp(rho_eq, rmin_eq, rhov)
    Rmaj_eq = (eqt1d.r_inboard .+ eqt1d.r_outboard) ./ 2
    return AlphaInput{T}(;
        rho=rhov,
        rmin=rminv,
        ne=_interp(rho_cp, ne, rhov),
        Te=_interp(rho_cp, Te, rhov),
        Ti=_interp(rho_cp, Ti, rhov),
        ni=_interp(rho_cp, ni, rhov),
        volume=_interp(rho_eq, vol_eq, rhov),
        Rmaj=_interp(rho_eq, Rmaj_eq, rhov),
        E_alpha=T(E_alpha), Z1=T(Z1), ln_lambda=T(ln_lambda))
end

# ──────────────────────────────────────────────────────────────────────────────
# public API
# ──────────────────────────────────────────────────────────────────────────────
"""
    run_alpha(dd, rho, crit_grad; solver=:stiff, method=:density, ep_mode=:fusion, kwargs...) -> AlphaResult
    run_alpha(input::AlphaInput, crit_grad; solver=:stiff, method=:density, ep_mode=:fusion) -> AlphaResult

Integrate the TGLF-EP critical gradients into energetic-particle profiles.

`solver`:
  * `:stiff` -- full stiff-CGM relaxation ([`stiff_cgm_transport`](@ref); Fortran
    `Alpha_transport.f90`). With `transport_params.use_ql_diffusivity=true`, adds
    [`ql_diffusivity!`](@ref) each iteration.
  * `:marginal` -- fast analytic marginal profile via [`integrate_crit_grad`](@ref)
    and `min(classical, marginal)`.

`ep_mode` (Fortran `NBI_flag`):
  * `:fusion` -- fusion alphas only (`NBI_flag=0`).
  * `:nbi` -- single NBI species (`NBI_flag=1`; uses `nbi`/`crit_grad` on primary slot).
  * `:fusion_nbi` -- fusion alphas + pencil-beam NBI (`NBI_flag=2`).

`method` (critical-gradient variable for the stiff threshold):
  * `:density` -- use `dndr_crit` from TJLFEP.
  * `:pressure` -- use `dpdr_crit` from TJLFEP.

`crit_grad` carries `dndr_crit` / `dpdr_crit` on the same `rho` grid as TJLFEP outputs.
For `:fusion_nbi`, optional `dndr_crit2` / `dpdr_crit2` for the NBI species.
"""
function run_alpha(dd::IMAS.dd, rho::AbstractVector, crit_grad; solver::Symbol=:stiff,
                   method::Symbol=:density, ep_mode::Symbol=:fusion,
                   transport_params=nothing, nbi=nothing, ql_modes=nothing,
                   E_alpha::Real=3.5, Z1::Real=5 // 3, ln_lambda::Real=17)
    input = AlphaInput(dd, rho; E_alpha, Z1, ln_lambda)
    return run_alpha(input, crit_grad; solver, method, ep_mode, transport_params, nbi, ql_modes)
end

function run_alpha(input::AlphaInput{T}, crit_grad; solver::Symbol=:stiff,
                   method::Symbol=:density, ep_mode::Symbol=:fusion,
                   transport_params=nothing,
                   nbi::Union{Nothing,NBIBeamParams{T}}=nothing,
                   ql_modes=nothing) where {T<:Real}
    n_cl, T_equiv, E_c_hat, S0 = slowing_down(input.ne, input.Te, input.Ti, input.ni;
        E_alpha=input.E_alpha, Z1=input.Z1, ln_lambda=input.ln_lambda)

    rmin = input.rmin

    n_EP2 = T[]
    n_He = T[]
    D_ql = T[]

    if solver === :stiff
        tp0 = transport_params === nothing ? AlphaTransportParams{T}() : transport_params
        i_tot = if ep_mode === :fusion_nbi && method === :pressure
            1
        elseif method === :density
            0
        else
            -1
        end
        tp = AlphaTransportParams{T}(;
            delta0=tp0.delta0, delta1=tp0.delta1, rdelta0=tp0.rdelta0,
            D_bkg=tp0.D_bkg, D_TAE=tp0.D_TAE, SDsink=tp0.SDsink,
            relax=tp0.relax, relax_f=tp0.relax_f, n_iter=tp0.n_iter, tol=tp0.tol,
            l_crit_smooth=tp0.l_crit_smooth, use_angioni_bkg=tp0.use_angioni_bkg,
            angioni_pinch_fac=tp0.angioni_pinch_fac, angioni_negative=tp0.angioni_negative,
            Q_fus=tp0.Q_fus, i_tot_TAE=i_tot,
            adapt_D_TAE=tp0.adapt_D_TAE, use_ql_diffusivity=tp0.use_ql_diffusivity,
            ql_params=tp0.ql_params, he_ash_params=tp0.he_ash_params)

        nbi_p = nbi === nothing ? NBIBeamParams{T}() : nbi
        n_cl2 = T_equiv2 = S02 = nothing
        if ep_mode === :fusion_nbi
            S02 = nbi_pencil_beam_source(input; nbi=nbi_p)
            Z1n = nbi_Z1(nbi_p.M_DT)
            n_cl2, T_equiv2, _ = slowing_down_nbi(input.ne, input.Te, S02;
                E_nbi=nbi_p.E_nbi, Z1_nbi=Z1n, ln_lambda=input.ln_lambda, M_DT=nbi_p.M_DT)
        elseif ep_mode === :nbi
            S0 = nbi_pencil_beam_source(input; nbi=nbi_p)
            n_cl, T_equiv, E_c_hat = slowing_down_nbi(input.ne, input.Te, S0;
                E_nbi=nbi_p.E_nbi, Z1_nbi=nbi_Z1(nbi_p.M_DT), ln_lambda=input.ln_lambda, M_DT=nbi_p.M_DT)
        end

        qmodes = if ql_modes === nothing
            dndr_q = _as_T(T, _getgrad(crit_grad, :dndr_crit))
            dndr_q === nothing ? QLModeInput{T}[] : default_ql_modes(dndr_q; params=tp.ql_params === nothing ? QLDiffusivityParams{T}() : tp.ql_params)
        else
            collect(QLModeInput{T}, ql_modes)
        end

        stiff = stiff_cgm_transport(input, n_cl, T_equiv, S0, crit_grad;
            params=tp, critgrad_method=method,
            n_cl2=n_cl2, T_equiv2=T_equiv2, S02=S02,
            crit_grad2=crit_grad, ql_modes=qmodes)
        n_EP = stiff.n_tran
        n_EP2 = stiff.n_tran2
        T_EP = [n_EP[i] > eps(T) ? stiff.p_tran[i] / (n_EP[i] * _KEV19_TO_KPA) : T_equiv[i] for i in eachindex(n_EP)]
        p_EP = n_EP .* T_EP
        flux_particle = stiff.flux
        transport_active = stiff.rg_p_tran .> stiff.rg_p_th .+ eps(T)
        D_alpha = stiff.D_alpha
        D_ql = stiff.D_ql
        stiff_error = stiff.error
        stiff_n_iter = stiff.n_iter
        if stiff.he_ash !== nothing
            n_He = stiff.he_ash.n_He
        end
    elseif solver === :marginal
        if method === :density
            dndr = _as_T(T, _getgrad(crit_grad, :dndr_crit))
            dndr === nothing && error("run_alpha(method=:density) requires `dndr_crit` in crit_grad")
            n_marg = integrate_crit_grad(rmin, dndr)
            n_EP = min.(n_cl, n_marg)
            T_EP = copy(T_equiv)
            p_EP = n_EP .* T_EP
        elseif method === :pressure
            dpdr = _as_T(T, _getgrad(crit_grad, :dpdr_crit))
            dpdr === nothing && error("run_alpha(method=:pressure) requires `dpdr_crit` in crit_grad")
            p_cl = n_cl .* T_equiv
            p_marg = integrate_crit_grad(rmin, dpdr)
            p_EP = min.(p_cl, p_marg)
            dndr = _as_T(T, _getgrad(crit_grad, :dndr_crit))
            n_EP = dndr === nothing ? copy(n_cl) : min.(n_cl, integrate_crit_grad(rmin, dndr))
            T_EP = [n_EP[i] > eps(T) ? p_EP[i] / n_EP[i] : T_equiv[i] for i in eachindex(n_EP)]
        else
            error("run_alpha: unknown method=$method")
        end
        transport_active = n_EP .< (n_cl .- eps(T) * 10)
        cumsrc = _cumtrapz(input.volume, S0)
        area = _dArea(input.volume, rmin)
        flux_particle = [area[i] > 0 ? cumsrc[i] / area[i] : zero(T) for i in eachindex(area)]
        D_alpha = zeros(T, length(n_EP))
        D_ql = zeros(T, length(n_EP))
        stiff_error = zero(T)
        stiff_n_iter = 0
    else
        error("run_alpha: unknown solver=$solver (use :stiff or :marginal)")
    end
    flux_energy = T(1.5) .* T_EP .* flux_particle

    return AlphaResult{T}(;
        rho=collect(T, input.rho), n_EP, p_EP, T_EP,
        flux_particle, flux_energy,
        n_classical=n_cl, T_alpha_equiv=T_equiv, E_c_hat, S0, transport_active,
        stiff_error, stiff_n_iter, D_alpha, D_ql, n_EP2, n_He)
end

_as_T(::Type{T}, ::Nothing) where {T} = nothing
_as_T(::Type{T}, v::AbstractVector) where {T} = collect(T, v)

end # module ALPHA
