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
  * `Alpha_transport.f90` (stiff-CGM relaxation) -> [`integrate_crit_grad`](@ref):
    the steady-state critical-gradient-model marginal profile that the stiff
    relaxation solver converges to (the actual EP profile is the lower of the
    source-limited classical profile and the transport-limited marginal profile).

The full time-dependent / quasilinear-diffusivity paths of the Fortran code are
out of scope (steady state only).
"""
module ALPHA

using LinearAlgebra
using Printf
import IMAS
import GACODE

export run_alpha, AlphaResult, AlphaInput

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
end

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
    return AlphaInput{T}(;
        rho=rhov,
        rmin=_interp(rho_eq, rmin_eq, rhov),
        ne=_interp(rho_cp, ne, rhov),
        Te=_interp(rho_cp, Te, rhov),
        Ti=_interp(rho_cp, Ti, rhov),
        ni=_interp(rho_cp, ni, rhov),
        volume=_interp(rho_eq, vol_eq, rhov),
        E_alpha=T(E_alpha), Z1=T(Z1), ln_lambda=T(ln_lambda))
end

# ──────────────────────────────────────────────────────────────────────────────
# public API
# ──────────────────────────────────────────────────────────────────────────────
"""
    run_alpha(dd, rho, crit_grad; method=:density, kwargs...) -> AlphaResult
    run_alpha(input::AlphaInput, crit_grad; method=:density) -> AlphaResult

Integrate the TGLF-EP critical gradients into energetic-particle profiles.

`crit_grad` provides the critical-gradient magnitude profiles on the same `rho`
grid; it may be a `NamedTuple`/struct (or `Dict`) carrying `dndr_crit` (critical
EP density gradient, 10^19 m^-3 per m) and/or `dpdr_crit` (critical EP pressure
gradient, 10^19 m^-3·keV per m) -- exactly the `dndr_crit_out` / `dpdr_crit_out`
returned by `TJLFEP.runTHD`.

`method`:
  * `:density`  -- integrate `dndr_crit` for the marginal density; the EP
    temperature is the slowing-down equivalent-Maxwellian `T_alpha_equiv` and
    `p_EP = n_EP · T_EP`.
  * `:pressure` -- integrate `dpdr_crit` for the marginal pressure; the EP
    temperature is `T_EP = p_EP / n_EP`.

In both cases the relaxed EP profile is the lower of the source-limited classical
slowing-down profile and the transport-limited marginal profile (critical-gradient
model). The EP particle flux is the steady-state volume-integrated source divided
by the flux surface; the energy flux is the convective `3/2 T_EP` estimate.
"""
function run_alpha(dd::IMAS.dd, rho::AbstractVector, crit_grad; method::Symbol=:density,
                   E_alpha::Real=3.5, Z1::Real=5 // 3, ln_lambda::Real=17)
    input = AlphaInput(dd, rho; E_alpha, Z1, ln_lambda)
    return run_alpha(input, crit_grad; method)
end

_getgrad(crit_grad, key) = crit_grad isa AbstractDict ? get(crit_grad, key, get(crit_grad, String(key), nothing)) :
                           (hasproperty(crit_grad, key) ? getproperty(crit_grad, key) : nothing)

function run_alpha(input::AlphaInput{T}, crit_grad; method::Symbol=:density) where {T<:Real}
    n_cl, T_equiv, E_c_hat, S0 = slowing_down(input.ne, input.Te, input.Ti, input.ni;
        E_alpha=input.E_alpha, Z1=input.Z1, ln_lambda=input.ln_lambda)

    rmin = input.rmin

    # marginal (transport-limited) profile from the critical gradient
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
        # density still from its own marginal if available, else classical
        dndr = _as_T(T, _getgrad(crit_grad, :dndr_crit))
        n_EP = dndr === nothing ? copy(n_cl) : min.(n_cl, integrate_crit_grad(rmin, dndr))
        T_EP = [n_EP[i] > 0 ? p_EP[i] / n_EP[i] : T_equiv[i] for i in eachindex(n_EP)]
    else
        error("run_alpha: unknown method=$method (use :density or :pressure)")
    end

    transport_active = n_EP .< (n_cl .- eps(T) * 10)

    # steady-state EP particle flux: Γ(r)·A(r) = ∫_0^r S0 dV, with A = dV/dr.
    cumsrc = _cumtrapz(input.volume, S0)         # ∫ S0 dV  [10^19 s^-1]
    area = _dArea(input.volume, rmin)            # dV/dr [m^2]
    flux_particle = [area[i] > 0 ? cumsrc[i] / area[i] : zero(T) for i in eachindex(area)]
    flux_energy = T(1.5) .* T_EP .* flux_particle

    return AlphaResult{T}(;
        rho=collect(T, input.rho), n_EP, p_EP, T_EP,
        flux_particle, flux_energy,
        n_classical=n_cl, T_alpha_equiv=T_equiv, E_c_hat, S0, transport_active)
end

_as_T(::Type{T}, ::Nothing) where {T} = nothing
_as_T(::Type{T}, v::AbstractVector) where {T} = collect(T, v)

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

end # module ALPHA
