# Readers for the Fortran GACODE `Alpha` run directories (Alpha.out / Alpha_transport.out /
# D_alpha.out / input.gacode / alpha_*_crit.input) and for the compact parity fixtures
# derived from them (test/fixtures/d3d_153071_*_alpha.txt). Used by the Fortran comparison
# harness (examples/fortran_comparison.jl) and the parity tests.

"""Read `n` consecutive single-float lines after `header` (dashed/units lines are skipped)."""
function _fortran_section(lines::Vector{String}, header::AbstractString, n::Int; occurrence::Int=1)
    hits = findall(l -> strip(l) == header, lines)
    length(hits) >= occurrence || error("section '$header' (occurrence $occurrence) not found")
    k = hits[occurrence] + 1
    vals = Float64[]
    while k <= length(lines) && length(vals) < n
        x = tryparse(Float64, strip(lines[k]))
        if x === nothing
            isempty(vals) || error("section '$header': non-numeric line inside data block: $(lines[k])")
        else
            push!(vals, x)
        end
        k += 1
    end
    length(vals) == n || error("section '$header': expected $n values, got $(length(vals))")
    return vals
end

"""Read a `# name | unit` block of input.gacode (rows `index value...`, first value column)."""
function _gacode_block(lines::Vector{String}, name::AbstractString, n::Int)
    k = findfirst(l -> startswith(l, "# ") && strip(first(split(l[3:end], "|"))) == name, lines)
    k === nothing && error("input.gacode block '$name' not found")
    vals = Float64[]
    for ln in @view lines[k+1:k+n]
        p = split(ln)
        push!(vals, parse(Float64, p[2]))
    end
    return vals
end

"""Parse the `ii= N D_TAE= x error=` / next-line iteration log into `Dict(ii => (error, error_f))`."""
function _fortran_iteration_log(lines::Vector{String})
    log = Dict{Int,Tuple{Float64,Float64}}()
    for (k, l) in enumerate(lines)
        m = match(r"^\s*ii=\s*(\d+)\s+D_TAE=\s*\S+\s+error=\s*$", l)
        m === nothing && continue
        p = split(lines[k+1])
        length(p) >= 2 || continue
        log[parse(Int, m.captures[1])] = (parse(Float64, p[1]), parse(Float64, p[2]))
    end
    return log
end

"""
    read_fortran_alpha_run(dir) -> NamedTuple

Parse a Fortran `Alpha` run directory (as produced by `Alpha_driver`) into the quantities
needed to re-run the stiff-CGM solve in Julia and compare against it. Double-precision
profiles are taken from `Alpha_transport.out`; geometry from `input.gacode`; critical
gradients from `alpha_dndr_crit.input` / `alpha_dpdr_crit.input` (raw file units).

Fields: `n`, `rho`, `rmin`, `kappa`, `Rmaj`, `Vp` (Fortran `V' = 4π²·κ·rmin·Rmaj`), `n_cl`,
`p_cl`, `T_equiv`, `S0`, `dndr` / `dpdr` (file values, `nothing` if absent), `rg_p_th`,
`n_tran`, `p_tran`, `fluxVp`, `D`, `error_final`, `n_iter`, `converged`, `errlog`, `params`
(echoed `i_tot_TAE`, `l_D_interface`, `l_norm_const`, `relax`, `n_up_loop`, `D_TAE`, `D_bkg`).
"""
function read_fortran_alpha_run(dir::AbstractString)
    tr = readlines(joinpath(dir, "Alpha_transport.out"))
    ga = readlines(joinpath(dir, "input.gacode"))
    n = parse(Int, first(split(strip(tr[findfirst(l -> occursin("n_rho_grid", l), tr)]))))
    getp(key) = begin
        k = findfirst(l -> occursin(key, l), tr)
        k === nothing ? nothing : parse(Float64, split(replace(tr[k], "=" => " "))[2])
    end
    params = (; i_tot_TAE=Int(getp("i_tot_TAE=")),
        l_D_interface=(x = getp("l_D_interface="); x === nothing ? 0 : Int(x)),
        l_norm_const=(x = getp("l_norm_const="); x === nothing ? 0 : Int(x)),
        relax=getp("relax="), n_up_loop=Int(getp("n_up_loop=")),
        D_TAE=getp("D_TAE ="), D_bkg=getp("D_bkg ="))

    rho = _fortran_section(tr, "r_hat or rho_hat grid", n)
    rmin = _gacode_block(ga, "rmin", n)
    kappa = _gacode_block(ga, "kappa", n)
    Rmaj = _gacode_block(ga, "rmaj", n)
    Vp = vprime_fortran(kappa, rmin, Rmaj)
    n_cl = _fortran_section(tr, "slowing down alpha density profile", n)
    p_cl = _fortran_section(tr, "slowing down alpha pressure profile", n)
    T_equiv = p_cl ./ (n_cl .* _KEV19_TO_KPA)
    S0 = _fortran_section(tr, "source", n)
    rg_p_th = _fortran_section(tr, "alpha critical pressure gradient profile", n)
    n_tran = _fortran_section(tr, "finished transported alpha density profile", n)
    p_tran = _fortran_section(tr, "finished transported alpha pressure profile", n)
    fluxVp = _fortran_section(tr, "finished flow of alpha density profile", n)
    D = _fortran_section(tr, "finished D_alpha", n)
    errlog = _fortran_iteration_log(tr)
    # final relative error: written to D_alpha.out (last line); fall back to the iteration log
    error_final = NaN
    if isfile(joinpath(dir, "D_alpha.out"))
        dl = readlines(joinpath(dir, "D_alpha.out"))
        kerr = findlast(l -> occursin("relative error in solution", l), dl)
        kerr === nothing || (error_final = parse(Float64, first(split(dl[kerr]))))
    end
    isnan(error_final) && !isempty(errlog) && (error_final = errlog[maximum(keys(errlog))][1])
    kconv = findfirst(l -> occursin("Converged at ii=", l) || occursin("Plateaued at ii=", l), tr)
    converged = kconv !== nothing
    n_iter = converged ? parse(Int, match(r"ii=\s*(\d+)", tr[kconv]).captures[1]) : maximum(keys(errlog))

    crit(f) = isfile(joinpath(dir, f)) ? read_crit_grad(joinpath(dir, f))[2] : nothing
    dndr = crit("alpha_dndr_crit.input")
    dpdr = crit("alpha_dpdr_crit.input")

    # self-checks against the single-precision echoes in Alpha.out and D_alpha.out
    if isfile(joinpath(dir, "Alpha.out"))
        ao = readlines(joinpath(dir, "Alpha.out"))
        chk(name, ref, hdr) = begin
            v = _fortran_section(ao, hdr, n)
            maximum(abs.(v .- ref) ./ max.(abs.(ref), 1e-12)) < 1e-5 ||
                error("read_fortran_alpha_run: $name in Alpha.out disagrees with Alpha_transport.out")
        end
        chk("n_cl", n_cl, "alpha density in 10**19 1/m**3")
        chk("S0", S0, "alpha source in 10**19 1/m**3/sec")
        chk("rmin", rmin, "midplane minor radius in meters")
        chk("kappa", kappa, "elongation")
    end
    if isfile(joinpath(dir, "D_alpha.out"))
        Dfile = _fortran_section(readlines(joinpath(dir, "D_alpha.out")), "Diffusion coefficient for EP species 1 (natural normalization) in m^2/s", n)
        Dfile == D || error("read_fortran_alpha_run: D_alpha.out differs from the 'finished D_alpha' section")
    end

    return (; n, rho, rmin, kappa, Rmaj, Vp, n_cl, p_cl, T_equiv, S0, dndr, dpdr, rg_p_th,
        n_tran, p_tran, fluxVp, D, error_final, n_iter, converged, errlog, params)
end

"""Fortran `Alpha` flux-surface area factor `V' = 2π κ rmin · 2π Rmaj` [m²]."""
vprime_fortran(kappa::AbstractVector, rmin::AbstractVector, Rmaj::AbstractVector) =
    4 * pi^2 .* kappa .* rmin .* Rmaj

const _ALPHA_FIXTURE_COLS = (:rho, :rmin, :kappa, :Rmaj, :n_cl, :T_equiv, :S0, :dndr, :dpdr,
    :rg_p_th_old, :rg_p_th_new, :D_old, :n_old, :fluxVp_old, :D_new, :n_new, :fluxVp_new)

"""
    write_alpha_fixture(path, old, new; header_lines=String[])

Write a compact parity fixture from an old-scheme and a new-scheme Fortran run of the same
case (both from [`read_fortran_alpha_run`](@ref)). Metadata lines are `# key: value`.
"""
function write_alpha_fixture(path::AbstractString, old, new; header_lines::Vector{String}=String[])
    old.n == new.n || error("old/new runs have different grids")
    for f in (:rho, :rmin, :kappa, :Rmaj, :n_cl, :S0)
        getfield(old, f) == getfield(new, f) || error("old/new runs differ in $f")
    end
    cols = (old.rho, old.rmin, old.kappa, old.Rmaj, old.n_cl, old.T_equiv, old.S0,
        old.dndr, new.dpdr, old.rg_p_th, new.rg_p_th, old.D, old.n_tran, old.fluxVp,
        new.D, new.n_tran, new.fluxVp)
    errline(run) = join(["$(k)=$(repr(run.errlog[k][1]))" for k in sort(collect(keys(run.errlog))) if k in (1, 2, 10, 100, 1000, 10000) || k == run.n_iter], " ")
    open(path, "w") do io
        for l in header_lines
            println(io, "# ", l)
        end
        println(io, "# old_n_iter: ", old.n_iter, "  old_error_final: ", repr(old.error_final), "  old_converged: ", old.converged)
        println(io, "# new_n_iter: ", new.n_iter, "  new_error_final: ", repr(new.error_final), "  new_converged: ", new.converged)
        println(io, "# old_errlog: ", errline(old))
        println(io, "# new_errlog: ", errline(new))
        println(io, "# columns: ", join(String.(_ALPHA_FIXTURE_COLS), " "))
        for i in 1:old.n
            println(io, join([@sprintf("%.16e", c[i]) for c in cols], " "))
        end
    end
    return path
end

"""
    read_alpha_fixture(path) -> NamedTuple

Read a fixture written by `write_alpha_fixture`: one field per column plus `meta`
(`old_n_iter`, `old_error_final`, `new_n_iter`, `new_error_final`, `old_errlog`, `new_errlog`
as `Dict{Int,Float64}`) and `Vp` (Fortran V').
"""
function read_alpha_fixture(path::AbstractString)
    cols = Dict(c => Float64[] for c in _ALPHA_FIXTURE_COLS)
    meta = Dict{Symbol,Any}()
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        if startswith(s, "#")
            for key in ("old_n_iter", "new_n_iter")
                m = match(Regex("$key:\\s*(\\d+)"), s)
                m === nothing || (meta[Symbol(key)] = parse(Int, m.captures[1]))
            end
            for key in ("old_error_final", "new_error_final")
                m = match(Regex("$key:\\s*(\\S+)"), s)
                m === nothing || (meta[Symbol(key)] = parse(Float64, m.captures[1]))
            end
            for key in ("old_errlog", "new_errlog")
                m = match(Regex("$key:\\s*(.*)\$"), s)
                m === nothing && continue
                meta[Symbol(key)] = Dict(parse(Int, split(t, "=")[1]) => parse(Float64, split(t, "=")[2]) for t in split(m.captures[1]))
            end
            continue
        end
        p = split(s)
        length(p) == length(_ALPHA_FIXTURE_COLS) || error("fixture $path: bad column count on line: $s")
        for (c, v) in zip(_ALPHA_FIXTURE_COLS, p)
            push!(cols[c], parse(Float64, v))
        end
    end
    nt = NamedTuple{_ALPHA_FIXTURE_COLS}(Tuple(cols[c] for c in _ALPHA_FIXTURE_COLS))
    return merge(nt, (; Vp=vprime_fortran(nt.kappa, nt.rmin, nt.Rmaj), meta=meta))
end
