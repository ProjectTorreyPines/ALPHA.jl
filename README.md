[![codecov](https://codecov.io/github/projecttorreypines/alpha.jl/graph/badge.svg?token=Osmi0iK6Vb)](https://codecov.io/github/projecttorreypines/alpha.jl)

# ALPHA.jl

Fast energetic-particle (EP) transport solver for TGLF-EP — a Julia port of the
steady-state GACODE `Alpha` model (E. Bass), including the 2026 stiff-transport fixes of
J. Lestz ([jlestz/Alpha](https://github.com/jlestz/Alpha)).

ALPHA takes the **critical-gradient profiles** produced by
[`TJLFEP`](https://github.com/ProjectTorreyPines/TJLFEP.jl) (`runTHD`) together
with the background plasma in an IMAS `dd`, and integrates those critical
gradients into the EP radial profiles — density `n_EP`, pressure `p_EP`,
temperature `T_EP = p_EP/n_EP` — plus the associated EP particle/energy flux.

## What it computes

| Stage | Function | Ported from |
|-------|----------|-------------|
| Classical slowing-down (density `n_classical`, equivalent-Maxwellian `T_alpha_equiv`, cross-over energy `E_c`) | `slowing_down` | `Alpha_comp_alpha_slowing.f90` |
| Marginal (transport-limited) profile from a critical gradient | `integrate_crit_grad` | analytic stiff-CGM limit |
| Stiff critical-gradient (CGM) flux-matching relaxation | `stiff_cgm_transport` | `Alpha_transport.f90` (incl. `l_D_interface`, `l_norm_const`, tolerance/plateau exit) |
| Quasi-linear EP diffusivity | `ql_diffusivity!` | QL-diffusivity coupling |
| Fusion + pencil-beam NBI dual-EP source | `nbi_pencil_beam_source`, `slowing_down_nbi` | NBI source model |
| Helium-ash transport | `he_ash_transport` | He ash model |

The top-level entry point is [`run_alpha`](#api), which orchestrates these.

## Units

All public profiles are in tokamak-transport units: densities `10^19 m^-3`,
temperatures `keV`, lengths `m`, pressures `10^19 m^-3 · keV`, particle flux
`10^19 m^-2 s^-1`, energy flux `keV · 10^19 m^-2 s^-1`.

The critical gradients are the exception: they are taken in the **TGLF-EP file units**, with
no conversion, exactly as the Fortran `Alpha` reads them — `dndr_crit` in `10^19 m^-3 / m`
(`alpha_dndr_crit.input`) and `dpdr_crit` in `10 kPa / m` (`alpha_dpdr_crit.input`; the
`dpdr_crit` returned by `TJLFEP.runTHD` is the same quantity). Hand the file values, or
`runTHD`'s output, straight to `run_alpha`; `load_crit_grad` reads the files. Internally the
stiff solver works in 10 kPa (`n·T·0.16022`) like the Fortran, so the pressure threshold it
applies is the file value itself. (ALPHA 1.x expected `dpdr_crit` divided by 0.16022 and, in
1.0, divided it again on the stiff path, making that threshold ~6× / ~39× too high.)

## Installation

ALPHA depends on `GACODE` and `IMAS` (registered in the
[FuseRegistry](https://github.com/ProjectTorreyPines/FuseRegistry.jl)) and
requires **Julia ≥ 1.11**.

```julia
using Pkg
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ProjectTorreyPines/FuseRegistry.jl.git"))
Pkg.Registry.add("General")
Pkg.add("ALPHA")
```

## Quick start (synthetic input, no TJLFEP)

`run_alpha` accepts an explicit `AlphaInput` and a `crit_grad` carrying
`dndr_crit` (density method) and/or `dpdr_crit` (pressure method) on the same
`rho` grid:

```julia
using ALPHA

n   = 51
rho = range(0.0, 1.0; length=n) |> collect
input = AlphaInput{Float64}(;
    rho,
    rmin   = 0.6 .* rho,                      # minor radius [m]
    ne     = 8.0 .* (1 .- 0.8 .* rho.^2),     # 10^19 m^-3
    Te     = 20.0 .* (1 .- 0.9 .* rho.^2) .+ 0.5,  # keV
    Ti     = 20.0 .* (1 .- 0.9 .* rho.^2) .+ 0.5,
    ni     = 7.2 .* (1 .- 0.8 .* rho.^2),
    volume = 30.0 .* rho.^2 .+ 1e-3,          # m^3
)

# critical density gradient from TJLFEP (here: a toy profile)
dndr_crit = [0.2 < r < 0.8 ? 1.0 : 5.0 for r in rho]

res = run_alpha(input, (; dndr_crit); solver=:stiff, method=:density)

res.n_EP            # transport-relaxed EP density
res.T_EP            # EP temperature
res.flux_particle   # EP particle flux
res.transport_active # where AE transport flattens the profile
```

## Full pipeline: TJLFEP → ALPHA on a sample equilibrium

The intended workflow couples TJLFEP and ALPHA on an IMAS `dd` built from a FUSE
sample case (e.g. `:ITER`, `:KDEMO`, `:D3D`). TJLFEP runs the TGLF-EP scan and
returns the critical gradients on the full radial grid; ALPHA integrates them.

A runnable script is provided in [`examples/tjlfep_alpha.jl`](examples/tjlfep_alpha.jl):

```julia
import FUSE, IMAS
using TJLFEP, ALPHA

# 1. Build a sample dd (ITER here; :KDEMO / :D3D also work)
ini, act = FUSE.case_parameters(:ITER; init_from=:ods)
ini.core_profiles.ngrid = 201
dd = IMAS.dd(); FUSE.init(dd, ini, act)

# 2. TJLFEP TGLF-EP scan -> critical gradients on the full rho grid
rho_scan = collect(range(0.2, 0.9; length=6))
OptionsDict = Dict{String,Any}(
    "nn"=>5, "nr"=>201, "jtscale_max"=>1, "nmodes"=>4, "PROCESS_IN"=>5,
    "THRESHOLD_FLAG"=>0, "N_BASIS"=>2, "SCAN_METHOD"=>1,
    "REJECT_I_PINCH_FLAG"=>0, "REJECT_E_PINCH_FLAG"=>0, "REJECT_TH_PINCH_FLAG"=>1,
    "REJECT_EP_PINCH_FLAG"=>0, "REJECT_TEARING_FLAG"=>1, "ROTATIONAL_SUPPRESSION_FLAG"=>1,
    "QL_RATIO_THRESH"=>0.001, "THETA_SQ_THRESH"=>100.0, "Q_SCALE"=>1.0,
    "WRITE_WAVEFUNCTION"=>0, "KY_MODEL"=>2, "SCAN_N"=>length(rho_scan), "IRS"=>2,
    "FACTOR_IN_PROFILE"=>false, "FACTOR_IN"=>1.0, "WIDTH_IN_FLAG"=>false,
    "WIDTH_MIN"=>1.0, "WIDTH_MAX"=>2.0, "INPUT_PROFILE_METHOD"=>2,
    "N_ION"=>3, "IS_EP"=>3, "REAL_FREQ"=>1)

_, _, SFmin, dpdr_crit, dndr_crit = runTHD(dd, rho_scan, OptionsDict; use_gpu=false)

# 3. ALPHA integrates the critical gradients into EP profiles
rho_full = collect(dd.core_profiles.profiles_1d[].grid.rho_tor_norm)
res = run_alpha(dd, rho_full, (; dndr_crit, dpdr_crit); solver=:stiff, method=:pressure)
```

`runTHD(dd, rho, OptionsDict)` returns
`(width, kymark, SFmin, dpdr_crit_out, dndr_crit_out, marginal_ql)`; the two
`*_crit_out` vectors live on the full `core_profiles` `rho_tor_norm` grid, which
is exactly the grid `run_alpha(dd, rho_full, …)` expects.

> **Separatrix note.** TJLFEP's scan always includes the last grid point
> `ir = NR` (`rho ≈ 1.0`). On a synthesized (FUSE) equilibrium the edge gradients
> there can be extreme enough to make the TGLF dispersion matrix singular; TJLFEP
> now treats such a point as stable (no AE mode) rather than erroring, so the
> scan completes and ALPHA receives a full-grid critical gradient.

<a name="api"></a>
## API

```julia
run_alpha(dd::IMAS.dd, rho, crit_grad; solver=:stiff, method=:pressure, ep_mode=:fusion, kwargs...)
run_alpha(input::AlphaInput, crit_grad; solver=:stiff, method=:pressure, ep_mode=:fusion)
```

- `solver`
  - `:stiff` — full stiff-CGM relaxation (`stiff_cgm_transport`); with
    `transport_params.use_ql_diffusivity=true` adds `ql_diffusivity!` each iteration.
  - `:marginal` — fast analytic marginal profile (`integrate_crit_grad`) and
    `min(classical, marginal)`.
- `method` — critical-gradient variable for the stiff threshold:
  - `:pressure` (default) — EP pressure-gradient drive against `dpdr_crit`, the
    TGLF-EP `alpha_dpdr_crit.input` threshold used as is (Fortran `i_tot_TAE=-1` with the
    dpdr file). This is the standard way to run Alpha.
  - `:density` — EP density-gradient drive against `dndr_crit` (Fortran `i_tot_TAE=0`),
    kept for comparison.
- `ep_mode` (Fortran `NBI_flag`): `:fusion` (alphas only), `:nbi` (single NBI
  species), `:fusion_nbi` (fusion alphas + pencil-beam NBI).
- `crit_grad` — a `NamedTuple`/`Dict` carrying `dndr_crit` / `dpdr_crit` (and
  optional `dndr_crit2` / `dpdr_crit2` for the second species) on the `rho` grid.

`AlphaInput(dd, rho; E_alpha=3.5, Z1=5/3, ln_lambda=17)` builds the background
plasma (ne, Te, Ti, ni, minor radius, volume, Rmaj) from an IMAS `dd` onto the
requested `rho_tor_norm` grid.

### Stiff-CGM solver options — `AlphaTransportParams`

Passed as `transport_params` to `run_alpha` (or `params` to `stiff_cgm_transport`). The
defaults reproduce the production Fortran `Alpha_transport.f90` including the 2026 fixes
of J. Lestz:

| field | default | Fortran | meaning |
|---|---|---|---|
| `D_interface` | `true` | `l_D_interface=1` | evaluate the stiff closure on the interface (half) grid from the one-sided gradient the recursion controls, and drive the recursion with that `D_half`; removes the grid-to-grid oscillation of `D` |
| `norm_const` | `true` | `l_norm_const=1` | normalise the supercritical excess by the grid maximum of the reference slowing-down profile instead of its local value; removes the edge diffusivity spike (a modelling change, not only a discretisation fix) |
| `n_iter` | `100000` | `n_up_loop` | maximum relaxation iterations |
| `tol` | `1e-12` | `error_tol` | early exit on the relative-change error |
| `plateau_window`, `plateau_ratio` | `10000`, `0.9` | `n_plateau_window`, `plateau_ratio_tol` | exit when the window-mean error stops decreasing (`plateau_window=0` disables) |
| `warn_nonconverged` | `true` | — | warn when `n_iter` is exhausted |
| `D_bkg`, `D_TAE`, `relax`, `relax_f` | `0.001`, `7.4`, `5e-4`, `0.005` | same | background/stiff diffusivities and relaxation factors |

The legacy (pre-1.1) closure is recovered with
`AlphaTransportParams(; D_interface=false, norm_const=false, tol=1e-3, n_iter=2000)`.
`stiff_cgm_transport` additionally accepts `Vp` (flux-surface area factor ``V' = dV/dr``
[m²]; `vprime_fortran(kappa, rmin, Rmaj)` gives the Fortran choice) and reports `D_half`,
`converged` and `exit_reason` (`:tol`, `:plateau`, `:max_iter`) in its result.

Together with feeding the TGLF-EP pressure threshold (`method=:pressure`, the
`alpha_dpdr_crit.input` values used directly), these options remove the three pathologies
found by J. Lestz in the original code. The axial diffusivity spike came from running the
pressure drive on a critical pressure gradient *derived* from `alpha_dndr_crit.input`
(`T·dndr·(1 + (ΔT/T)(n/Δn))`), which turns strongly negative where the slowing-down
density is flat or hollow; supplying the dpdr file bypasses that translation. ALPHA.jl only
performs the translation inside `method=:density`, where the stiff drive does not use it.

### Output — `AlphaResult`

`rho`, `n_EP`, `p_EP`, `T_EP`, `flux_particle`, `flux_energy`, plus diagnostics
`n_classical`, `T_alpha_equiv`, `E_c_hat`, `S0`, `transport_active`, the
stiff-CGM `stiff_error` / `stiff_n_iter` / `stiff_converged` / `stiff_exit_reason` /
`D_alpha` / `D_ql`, and the optional second-species `n_EP2` (NBI) and `n_He` (helium ash).

## Fortran parity

`examples/fortran_comparison.jl` re-runs the stiff solver on the inputs of Jeff Lestz's
reference Fortran `Alpha` runs (DIII-D 153071, t = 3400 ms "axial blow-up" and t = 2200 ms
"edge blow-up"; parsed with `read_fortran_alpha_run`, frozen in
`test/fixtures/d3d_153071_*_alpha.txt`) and compares against both the pre-fix and the fixed
Fortran output. With the legacy flags ALPHA.jl reproduces the old Fortran to ~1e-10
(including the 463 m²/s axial and 7.4 m²/s edge spikes); with the defaults it reproduces
the fixed Fortran to ~1e-11 and exits within 0.2 % of the same iteration count. The script
also prints the {threshold} × {`D_interface`} × {`norm_const`} attribution matrix and,
when `Plots` is available, writes a comparison figure. These checks run in the test suite
from the committed fixtures.

## Tests

```bash
julia --project=. test/runtests.jl
```

The suite includes golden-value tests of the legacy closure, closed-form checks of the
interface/max-normalised closures, convergence-exit tests, the `dpdr_crit` unit
convention, and the Fortran parity tests above.
