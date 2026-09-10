#!/usr/bin/env julia
# Compare ALPHA.jl's stiff-CGM solver against the Fortran `Alpha` reference runs of
# Jeff Lestz (DIII-D 153071, t=3400 ms "axial blow-up" and t=2200 ms "edge blow-up"):
#   old scheme  (pre-fix Fortran: dndr file, point closure, local normalisation, 10000 iters)
#   new scheme  (github.com/jlestz/Alpha: dpdr file, l_D_interface=1, l_norm_const=1, tol 1e-12)
# plus the full 2x2x2 attribution matrix {dndr,dpdr} x {D_interface} x {norm_const}.
#
# Inputs: the committed fixtures test/fixtures/d3d_153071_{3400,2200}_alpha.txt, or the live
# reference directories when ALPHA_FORTRAN_REF_DIR (default /global/cfs/cdirs/m3739/TGLFEP/jlestz)
# exists. Output: a Markdown table on stdout and, if Plots is available in the active
# environment, alpha_fortran_comparison_153071.png in ALPHA_COMPARISON_OUT (default: a temp dir).
using ALPHA
using Printf

const REF = get(ENV, "ALPHA_FORTRAN_REF_DIR", "/global/cfs/cdirs/m3739/TGLFEP/jlestz")
const OUT = get(ENV, "ALPHA_COMPARISON_OUT", mktempdir())
const FIXDIR = joinpath(@__DIR__, "..", "test", "fixtures")

"""Load one case either from the live Fortran directories or from the committed fixture."""
function load_case(t::AbstractString)
    dold = joinpath(REF, "153071E42_$(t)_fix2_100")
    dnew = joinpath(REF, "153071E42_$(t)_fix3_100")
    if isdir(dold) && isdir(dnew)
        old = ALPHA.read_fortran_alpha_run(dold)
        new = ALPHA.read_fortran_alpha_run(dnew)
        return (; rho=old.rho, rmin=old.rmin, Rmaj=old.Rmaj, Vp=old.Vp, n_cl=old.n_cl, T_equiv=old.T_equiv, S0=old.S0,
            dndr=old.dndr, dpdr=new.dpdr, rg_p_th_old=old.rg_p_th, rg_p_th_new=new.rg_p_th,
            D_old=old.D, n_old=old.n_tran, fluxVp_old=old.fluxVp, D_new=new.D, n_new=new.n_tran, fluxVp_new=new.fluxVp,
            meta=Dict(:old_n_iter => old.n_iter, :new_n_iter => new.n_iter, :old_error_final => old.error_final,
                :new_error_final => new.error_final, :old_errlog => Dict(k => v[1] for (k, v) in old.errlog),
                :new_errlog => Dict(k => v[1] for (k, v) in new.errlog)), source="live: $REF")
    else
        fx = ALPHA.read_alpha_fixture(joinpath(FIXDIR, "d3d_153071_$(t)_alpha.txt"))
        return merge(fx, (; source="fixture"))
    end
end

reldev(x, ref; floor=1e-3 * maximum(abs.(ref))) = maximum(abs.(x .- ref) ./ max.(abs.(ref), floor))
Ddev(x, ref) = maximum(abs.(x .- ref) ./ max.(ref, 1e-3))
"""Grid-to-grid oscillation metric: Σ|D[i+1]-2D[i]+D[i-1]| / Σ|D[i+1]-D[i]| (0 smooth … 2 zig-zag)."""
osc(D) = (s = sum(abs.(diff(D))); s > 0 ? sum(abs.(D[3:end] .- 2 .* D[2:end-1] .+ D[1:end-2])) / s : 0.0)
trapz(x, y) = sum(0.5 .* (y[2:end] .+ y[1:end-1]) .* diff(x))

const COMMON = (; D_bkg=0.001, D_TAE=7.4, relax=5e-4, relax_f=0.005, delta0=0.01, delta1=0.0, rdelta0=0.5,
    SDsink=1.0, i_tot_TAE=-1, plateau_window=0, warn_nonconverged=false)

function solve(c, thr::Symbol, D_interface::Bool, norm_const::Bool; n_iter=100_000, tol=1e-12, plateau_window=0)
    n = length(c.rho)
    input = ALPHA.AlphaInput{Float64}(; rho=c.rho, rmin=c.rmin, Rmaj=c.Rmaj,
        ne=ones(n), Te=ones(n), Ti=ones(n), ni=ones(n), volume=zeros(n))
    params = ALPHA.AlphaTransportParams{Float64}(; COMMON..., D_interface, norm_const, n_iter, tol, plateau_window)
    crit = thr === :density ? (; dndr_crit=min.(c.dndr, 1000.0)) : (; dpdr_crit=min.(c.dpdr, 1000.0) ./ 0.16022)
    return ALPHA.stiff_cgm_transport(input, c.n_cl, c.T_equiv, c.S0, crit; params, critgrad_method=thr, Vp=c.Vp)
end

function metrics(c, D, nEP, name; n_iter=0, exit_reason=:fortran, err=NaN)
    rho = c.rho
    Daxis = maximum(D[rho.<0.05]); Dcore = maximum(D[0.05 .<= rho .<= 0.9]); Dedge = maximum(D[rho.>0.9])
    ipk = argmax(nEP)
    Ntot = trapz(c.rmin, nEP .* c.Vp)
    return (; name, Daxis, Dcore, Dedge, osc=osc(D), n_iter, exit_reason, err, n_peak=nEP[ipk], rho_peak=rho[ipk], Ntot)
end

results = Dict{String,Any}()
for t in ("3400", "2200")
    c = load_case(t)
    println("\n## DIII-D 153071 t=$(t) ms  (reference: $(c.source))\n")
    rows = Any[]
    push!(rows, metrics(c, c.D_old, c.n_old, "Fortran old"; n_iter=c.meta[:old_n_iter], err=c.meta[:old_error_final]))
    push!(rows, metrics(c, c.D_new, c.n_new, "Fortran new"; n_iter=c.meta[:new_n_iter], err=c.meta[:new_error_final]))
    # parity runs
    so = solve(c, :density, false, false; n_iter=c.meta[:old_n_iter], tol=0.0)
    sn = solve(c, :pressure, true, true)
    push!(rows, metrics(c, so.D_alpha, so.n_tran, "Julia old scheme"; n_iter=so.n_iter, exit_reason=so.exit_reason, err=so.error))
    push!(rows, metrics(c, sn.D_alpha, sn.n_tran, "Julia new scheme"; n_iter=sn.n_iter, exit_reason=sn.exit_reason, err=sn.error))
    # attribution matrix
    runs = Dict{Tuple{Symbol,Bool,Bool},Any}()
    for thr in (:density, :pressure), di in (false, true), nc in (false, true)
        s = solve(c, thr, di, nc)
        runs[(thr, di, nc)] = s
        (thr, di, nc) in ((:density, false, false), (:pressure, true, true)) && continue
        push!(rows, metrics(c, s.D_alpha, s.n_tran, "Julia $(thr === :density ? "dndr" : "dpdr") D_interface=$(Int(di)) norm_const=$(Int(nc))";
            n_iter=s.n_iter, exit_reason=s.exit_reason, err=s.error))
    end
    sp = solve(c, :pressure, true, true; plateau_window=10_000)
    push!(rows, metrics(c, sp.D_alpha, sp.n_tran, "Julia new scheme + plateau exit"; n_iter=sp.n_iter, exit_reason=sp.exit_reason, err=sp.error))

    println("| run | D max ρ<0.05 | D max core | D max ρ>0.9 | osc(D) | n_iter | exit | final error | n peak (ρ) | N_tot |")
    println("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows
        @printf("| %s | %.4g | %.4g | %.4g | %.3f | %d | %s | %.3e | %.4g (%.2f) | %.4g |\n",
            r.name, r.Daxis, r.Dcore, r.Dedge, r.osc, r.n_iter, r.exit_reason, r.err, r.n_peak, r.rho_peak, r.Ntot)
    end
    println("\nParity (max relative deviation Julia vs Fortran):")
    @printf("  old: rg_p_th %.2e  n %.2e  D %.2e  flux·V' %.2e  final error %.6e vs %.6e  (ii=1 error %.15g vs %.15g)\n",
        reldev(so.rg_p_th, c.rg_p_th_old), reldev(so.n_tran, c.n_old), Ddev(so.D_alpha, c.D_old), reldev(so.flux .* c.Vp, c.fluxVp_old),
        so.error, c.meta[:old_error_final], solve(c, :density, false, false; n_iter=1, tol=0.0).error, c.meta[:old_errlog][1])
    @printf("  new: rg_p_th %.2e  n %.2e  D %.2e  flux·V' %.2e  exit iteration %d vs %d  (ii=1 error %.15g vs %.15g)\n",
        reldev(sn.rg_p_th, c.rg_p_th_new), reldev(sn.n_tran, c.n_new), Ddev(sn.D_alpha, c.D_new), reldev(sn.flux .* c.Vp, c.fluxVp_new),
        sn.n_iter, c.meta[:new_n_iter], solve(c, :pressure, true, true; n_iter=1, tol=0.0).error, c.meta[:new_errlog][1])
    results[t] = (; c, so, sn, runs)
end

# ---- optional figure ----
haveplots = try
    @eval using Plots
    true
catch
    @warn "Plots not available in this environment; table only"
    false
end
if haveplots
    plt = plot(; layout=(2, 2), size=(1100, 800), left_margin=5Plots.mm, bottom_margin=4Plots.mm)
    for (k, t) in enumerate(("3400", "2200"))
        r = results[t]; c = r.c
        plot!(plt[2k-1], c.rho, max.(c.D_old, 1e-4); seriestype=:scatter, ms=3, msw=0, label="Fortran old", yscale=:log10, color=:gray)
        plot!(plt[2k-1], c.rho, max.(c.D_new, 1e-4); seriestype=:scatter, ms=3, msw=0, label="Fortran new (jlestz)", color=:black)
        plot!(plt[2k-1], c.rho, max.(r.so.D_alpha, 1e-4); label="ALPHA.jl old scheme", color=:red, lw=1.5)
        plot!(plt[2k-1], c.rho, max.(r.sn.D_alpha, 1e-4); label="ALPHA.jl new scheme", color=:blue, lw=1.5)
        plot!(plt[2k-1], c.rho, max.(r.runs[(:pressure, false, false)].D_alpha, 1e-4); label="dpdr only", color=:green, ls=:dash)
        plot!(plt[2k-1], c.rho, max.(r.runs[(:pressure, true, false)].D_alpha, 1e-4); label="dpdr + D_interface", color=:orange, ls=:dash)
        plot!(plt[2k-1], c.rho, max.(r.runs[(:pressure, false, true)].D_alpha, 1e-4); label="dpdr + norm_const", color=:purple, ls=:dash)
        plot!(plt[2k-1]; title="153071 t=$(t) ms: D_EP [m²/s]", xlabel="ρ", legend=(k == 1 ? :topright : false))
        plot!(plt[2k], c.rho, c.n_cl; label="slowing-down n_cl", color=:gray, ls=:dot)
        plot!(plt[2k], c.rho, c.n_old; seriestype=:scatter, ms=3, msw=0, label="Fortran old", color=:gray)
        plot!(plt[2k], c.rho, c.n_new; seriestype=:scatter, ms=3, msw=0, label="Fortran new (jlestz)", color=:black)
        plot!(plt[2k], c.rho, r.so.n_tran; label="ALPHA.jl old scheme", color=:red, lw=1.5)
        plot!(plt[2k], c.rho, r.sn.n_tran; label="ALPHA.jl new scheme", color=:blue, lw=1.5)
        plot!(plt[2k]; title="n_EP [10¹⁹ m⁻³]", xlabel="ρ", legend=(k == 1 ? :topright : false))
    end
    pngfile = joinpath(OUT, "alpha_fortran_comparison_153071.png")
    savefig(plt, pngfile)
    println("\nfigure written to $pngfile")
end
