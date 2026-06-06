#!/usr/bin/env julia
# TJLFEP -> ALPHA on a FUSE sample equilibrium (ITER by default; set ALPHA_CASE).
#
# Requires an environment with BOTH TJLFEP and ALPHA (plus FUSE/IMAS) available,
# e.g. a project that `dev`s the sibling packages:
#   julia --project=<env> examples/tjlfep_alpha.jl
#
# Env knobs: ALPHA_CASE (ITER|KDEMO|D3D), SCAN_N (number of scan radii).
ENV["TJLFEP_FILE_ONLY"] = "0"

import FUSE
import IMAS
using TJLFEP
using ALPHA
using Printf

const CASE   = Symbol(get(ENV, "ALPHA_CASE", "ITER"))
const SCAN_N = parse(Int, get(ENV, "SCAN_N", "6"))

# 1. Build a sample dd ------------------------------------------------------------
println("=== building $CASE dd via FUSE ===")
ini, act = FUSE.case_parameters(CASE; init_from=:ods)
ini.core_profiles.ngrid = 201
dd = IMAS.dd()
FUSE.init(dd, ini, act)

# 2. TJLFEP TGLF-EP scan -> critical gradients ------------------------------------
rho_scan = collect(range(0.2, 0.9; length=SCAN_N))
OptionsDict = Dict{String,Any}(
    "nn"=>5, "nr"=>201, "jtscale_max"=>1, "nmodes"=>4, "PROCESS_IN"=>5,
    "THRESHOLD_FLAG"=>0, "N_BASIS"=>2, "SCAN_METHOD"=>1,
    "REJECT_I_PINCH_FLAG"=>0, "REJECT_E_PINCH_FLAG"=>0, "REJECT_TH_PINCH_FLAG"=>1,
    "REJECT_EP_PINCH_FLAG"=>0, "REJECT_TEARING_FLAG"=>1, "ROTATIONAL_SUPPRESSION_FLAG"=>1,
    "QL_RATIO_THRESH"=>0.001, "THETA_SQ_THRESH"=>100.0, "Q_SCALE"=>1.0,
    "WRITE_WAVEFUNCTION"=>0, "KY_MODEL"=>2, "SCAN_N"=>SCAN_N, "IRS"=>2,
    "FACTOR_IN_PROFILE"=>false, "FACTOR_IN"=>1.0, "WIDTH_IN_FLAG"=>false,
    "WIDTH_MIN"=>1.0, "WIDTH_MAX"=>2.0, "INPUT_PROFILE_METHOD"=>2,
    "N_ION"=>3, "IS_EP"=>3, "REAL_FREQ"=>1)

println("=== TJLFEP runTHD scan (CPU) ===")
outdir = mktempdir()
width, kymark, SFmin, dpdr_crit, dndr_crit = cd(outdir) do
    runTHD(dd, rho_scan, OptionsDict; printout=true, saveFiles=false, dir=outdir, use_gpu=false)
end
@printf("TJLFEP done. SFmin = %s\n", SFmin)

# 3. ALPHA integrates the critical gradients into EP profiles ---------------------
println("=== ALPHA run_alpha ===")
rho_full = collect(dd.core_profiles.profiles_1d[].grid.rho_tor_norm)
res = run_alpha(dd, rho_full, (; dndr_crit, dpdr_crit); solver=:stiff, method=:density)

@printf("ALPHA done. stiff_n_iter=%d stiff_error=%.3g\n", res.stiff_n_iter, res.stiff_error)
@printf("  n_EP    : min=%.4g max=%.4g [10^19 m^-3]\n", minimum(res.n_EP), maximum(res.n_EP))
@printf("  T_EP    : min=%.4g max=%.4g [keV]\n", minimum(res.T_EP), maximum(res.T_EP))
@printf("  AE-limited radii: %d / %d\n", count(res.transport_active), length(res.transport_active))
println("=== done ===")
