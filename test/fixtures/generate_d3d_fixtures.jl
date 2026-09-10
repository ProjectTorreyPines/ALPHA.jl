#!/usr/bin/env julia
# Generate the committed Fortran-parity fixtures test/fixtures/d3d_153071_{3400,2200}_alpha.txt
# from Jeff Lestz's reference Fortran `Alpha` runs (old = pre-fix code, new = github.com/jlestz/Alpha
# with l_D_interface=1, l_norm_const=1 and the pressure critical-gradient file).
#
# Needs read access to the reference directories (NERSC CFS, project m3739):
#   ALPHA_FORTRAN_REF_DIR (default /global/cfs/cdirs/m3739/TGLFEP/jlestz)
# Run from an environment with ALPHA available:
#   julia --project=<env> test/fixtures/generate_d3d_fixtures.jl
using ALPHA
using Printf

const REF = get(ENV, "ALPHA_FORTRAN_REF_DIR", "/global/cfs/cdirs/m3739/TGLFEP/jlestz")
isdir(REF) || error("reference directory $REF not found; set ALPHA_FORTRAN_REF_DIR")

const CASES = (("3400", "axial diffusivity blow-up in the old code"),
               ("2200", "edge diffusivity blow-up in the old code"))

for (t, why) in CASES
    old = ALPHA.read_fortran_alpha_run(joinpath(REF, "153071E42_$(t)_fix2_100"))
    new = ALPHA.read_fortran_alpha_run(joinpath(REF, "153071E42_$(t)_fix3_100"))
    header = [
        "Fortran `Alpha` (GACODE, E. Bass) parity fixture: DIII-D 153071 t=$(t) ms, 80 keV D NBI (TRANSP BDEP source), 101-point rho grid.",
        "Reference runs by Jeff Lestz (jlestz@pppl.gov): $(REF)/153071E42_$(t)_fix{2,3}_100 ($why).",
        "old = pre-fix Alpha (2026-08-16): alpha_dndr_crit.input, i_tot_TAE=-1, n_up_loop=10000, no tolerance exit (unconverged).",
        "new = github.com/jlestz/Alpha (2026-09-06, between bf263a6 and 3600340): alpha_dpdr_crit.input, l_D_interface=1, l_norm_const=1, error_tol=1e-12, n_up_loop=100000, no plateau exit.",
        "Common parameters: D_bkg=0.001 D_TAE=7.4 relax=5e-4 relax_f=0.005 delta0=0.01 delta1=0 rdelta0=0.5 SDsink=1 l_crit_smooth=1 i_tot_TAE=-1; a=rmin[end]; V'=4*pi^2*kappa*rmin*Rmaj;",
        "critical gradients are the raw TGLF-EP file values (dndr: 10^19 m^-3/m, dpdr: 10 kPa/m), capped by the Fortran composite min(file,1000).",
        "Units: rho[-] rmin,Rmaj[m] kappa[-] n[10^19 m^-3] T_equiv[keV] S0[10^19 m^-3/s] rg_p_th[10 kPa/m] D[m^2/s] fluxVp[10^19 m^-3 m/s m^2]",
    ]
    path = joinpath(@__DIR__, "d3d_153071_$(t)_alpha.txt")
    ALPHA.write_alpha_fixture(path, old, new; header_lines=header)
    fx = ALPHA.read_alpha_fixture(path)
    @assert fx.n_old == old.n_tran && fx.D_new == new.D
    @printf("wrote %s (%d rows; old n_iter=%d err=%.3e, new n_iter=%d err=%.3e)\n", path, length(fx.rho),
        fx.meta[:old_n_iter], fx.meta[:old_error_final], fx.meta[:new_n_iter], fx.meta[:new_error_final])
end
