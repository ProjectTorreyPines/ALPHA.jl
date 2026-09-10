using Test
using ALPHA

@testset "ALPHA.jl" begin

    @testset "DT_sigma_v asset" begin
        Tgrid, sigv = ALPHA.load_DT_sigma_v()
        @test length(Tgrid) == length(sigv)
        @test length(Tgrid) > 100
        @test issorted(Tgrid)
        @test all(>=(0), sigv)
        # reactivity rises with temperature in the 1-20 keV range
        i10 = findmin(abs.(Tgrid .- 10.0))[2]
        i2 = findmin(abs.(Tgrid .- 2.0))[2]
        @test sigv[i10] > sigv[i2]
    end

    @testset "slowing_down physics" begin
        n = 51
        ne = fill(8.0, n)        # 10^19 m^-3
        Te = range(20.0, 1.0; length=n) |> collect
        Ti = copy(Te)
        ni = fill(7.0, n)
        n_cl, T_eq, E_c, S0 = ALPHA.slowing_down(ne, Te, Ti, ni; E_alpha=3.5, Z1=5 / 3, ln_lambda=17.0)
        @test all(>(0), n_cl)
        @test all(>(0), S0)
        # equivalent Maxwellian temperature is a fraction of the birth energy (3.5 MeV = 3500 keV)
        @test all(0 .< T_eq .< 3500)
        # hotter core => higher cross-over energy than cooler edge
        @test E_c[1] > E_c[end]
    end

    @testset "integrate_crit_grad (constant gradient -> linear profile)" begin
        rmin = range(0.0, 1.0; length=11) |> collect
        g = fill(2.0, 11)                       # -dX/dr = 2 everywhere
        X = ALPHA.integrate_crit_grad(rmin, g; X_edge=0.0)
        @test X[end] ≈ 0.0 atol = 1e-12
        @test X[1] ≈ 2.0 atol = 1e-10          # integral of 2 over length 1
        @test issorted(X; rev=true)            # decreasing outward
    end

    @testset "run_alpha (AlphaInput path)" begin
        n = 51
        rho = range(0.0, 1.0; length=n) |> collect
        rmin = 0.6 .* rho
        ne = 8.0 .* (1 .- 0.8 .* rho .^ 2)
        Te = 20.0 .* (1 .- 0.9 .* rho .^ 2) .+ 0.5
        Ti = copy(Te)
        ni = 0.9 .* ne
        volume = 30.0 .* rho .^ 2 .+ 1e-3
        input = ALPHA.AlphaInput{Float64}(; rho, rmin, ne, Te, Ti, ni, volume)

        # critical density gradient that turns on in the mid-radius (AE drive region)
        dndr_crit = [0.2 < r < 0.8 ? 1.0 : 5.0 for r in rho]
        res = run_alpha(input, (; dndr_crit); solver=:marginal, method=:density)

        @test length(res.n_EP) == n
        @test all(res.n_EP .>= 0)
        @test all(res.n_EP .<= res.n_classical .+ 1e-9)   # transport can only flatten
        @test all(isfinite, res.flux_particle)
        @test all(isfinite, res.flux_energy)
        @test all(res.T_EP .≈ res.T_alpha_equiv)          # :density => T from slowing-down
        @test any(res.transport_active)                    # some region is AE-limited
        @test res.stiff_n_iter == 0
        # pressure method (marginal)
        dpdr_crit = dndr_crit .* res.T_alpha_equiv
        res2 = run_alpha(input, (; dndr_crit, dpdr_crit); solver=:marginal, method=:pressure)
        @test all(res2.p_EP .>= 0)
        @test all(isfinite, res2.T_EP)
    end

    @testset "stiff_cgm_transport" begin
        n = 31
        rho = range(0.0, 1.0; length=n) |> collect
        rmin = 0.6 .* rho
        ne = 8.0 .* (1 .- 0.8 .* rho .^ 2)
        Te = 20.0 .* (1 .- 0.9 .* rho .^ 2) .+ 0.5
        Ti = copy(Te)
        ni = 0.9 .* ne
        volume = 30.0 .* rho .^ 2 .+ 1e-3
        input = ALPHA.AlphaInput{Float64}(; rho, rmin, ne, Te, Ti, ni, volume)
        n_cl, T_eq, _, S0 = ALPHA.slowing_down(ne, Te, Ti, ni; E_alpha=3.5, Z1=5 / 3, ln_lambda=17.0)
        dndr_crit = [0.2 < r < 0.8 ? 0.5 : 2.0 for r in rho]
        params = ALPHA.AlphaTransportParams{Float64}(; n_iter=500, tol=1e-2, relax=1e-3, warn_nonconverged=false)
        stiff = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit);
            params=params, critgrad_method=:density)
        @test stiff.n_iter > 0
        @test stiff.error < 1.0
        @test all(>=(0), stiff.n_tran)
        @test all(isfinite, stiff.flux)
        @test maximum(stiff.D_alpha) >= params.D_bkg

        res = run_alpha(input, (; dndr_crit); solver=:stiff, method=:density,
            transport_params=params)
        @test res.stiff_n_iter > 0
        @test res.stiff_error < 1.0
        @test all(res.n_EP .>= 0)
        @test all(isfinite, res.flux_particle)
    end

    @testset "ql_diffusivity (stiff-CGM coupling)" begin
        n = 21
        rho = range(0.0, 1.0; length=n) |> collect
        rmin = 0.6 .* rho
        input = ALPHA.AlphaInput{Float64}(;
            rho, rmin,
            ne=fill(8.0, n), Te=fill(15.0, n), Ti=fill(15.0, n), ni=fill(7.0, n),
            volume=30.0 .* rho .^ 2, Rmaj=fill(6.2, n))
        dndr = fill(0.5, n)
        params = ALPHA.QLDiffusivityParams{Float64}(; km_max=3, dt_update=1e-2)
        tp = ALPHA.AlphaTransportParams{Float64}(;
            n_iter=300, tol=1e-2, relax=1e-3, use_ql_diffusivity=true, ql_params=params, warn_nonconverged=false)
        res = run_alpha(input, (; dndr_crit=dndr); solver=:stiff, transport_params=tp)
        @test length(res.D_ql) == n
        @test all(isfinite, res.D_ql)
        @test res.stiff_n_iter > 0
    end

    @testset "ITER (FUSE + TJLFEP crit grads) regression" begin
        # Frozen fixture: FUSE :ITER background profiles (as AlphaInput(dd, rho) extracts
        # them) + TJLFEP.runTHD critical gradients on the full rho grid. Regenerate with
        # test/fixtures/generate_iter_fixture.jl (needs FUSE + TJLFEP). Because the inputs
        # are frozen, the :marginal outputs below are deterministic golden values.
        fixture = joinpath(@__DIR__, "fixtures", "iter_tjlfep.txt")
        cols = [Float64[] for _ in 1:10]
        for line in eachline(fixture)
            (isempty(line) || startswith(line, "#")) && continue
            p = split(line)
            for j in 1:10
                push!(cols[j], parse(Float64, p[j]))
            end
        end
        rho, rmin, ne, Te, Ti, ni, volume, Rmaj, dndr_crit, dpdr_crit = cols
        dpdr_crit = dpdr_crit ./ 0.16022   # fixture stores the raw TJLFEP value (10 kPa/m); ALPHA wants 10^19 m^-3·keV/m
        @test length(rho) == 201
        input = ALPHA.AlphaInput{Float64}(; rho, rmin, ne, Te, Ti, ni, volume, Rmaj)

        # marginal solver: analytic transport-limited profile -> deterministic regression
        res = run_alpha(input, (; dndr_crit, dpdr_crit); solver=:marginal, method=:density)
        @test all(isfinite, res.n_EP)
        @test all(>=(0), res.n_EP)
        @test all(res.n_EP .<= res.n_classical .+ 1e-9)   # transport can only flatten
        @test all(isfinite, res.flux_particle)
        @test all(isfinite, res.flux_energy)
        @test all(isfinite, res.T_EP)
        @test any(res.transport_active)                    # ITER edge crit-grad flattens somewhere
        # golden values (inputs frozen => deterministic, pure-arithmetic marginal path)
        @test maximum(res.n_EP) ≈ 0.1053084649712 rtol = 1e-6
        @test sum(res.n_EP) ≈ 6.472034365065 rtol = 1e-6
        @test res.n_EP[101] ≈ 0.01475189673948 rtol = 1e-6
        @test maximum(res.T_EP) ≈ 998.8427884231 rtol = 1e-6
        @test sum(res.p_EP) ≈ 6248.616489435 rtol = 1e-6

        # pressure method (marginal)
        resp = run_alpha(input, (; dndr_crit, dpdr_crit); solver=:marginal, method=:pressure)
        @test all(>=(0), resp.p_EP)
        @test all(isfinite, resp.T_EP)
        @test sum(resp.p_EP) ≈ 6248.616489435 rtol = 1e-6

        # stiff solver: exercise the full relaxation path on real ITER data (finite, physical)
        ress = run_alpha(input, (; dndr_crit, dpdr_crit); solver=:stiff, method=:density)
        @test all(isfinite, ress.n_EP)
        @test all(>=(0), ress.n_EP)
        @test all(isfinite, ress.flux_particle)
        @test ress.stiff_converged
        @test ress.stiff_exit_reason in (:tol, :plateau)
    end


    # ------------------------------------------------------------------------------------
    # Stiff-CGM closure options (port of J. Lestz's Alpha_transport.f90 fixes)
    # ------------------------------------------------------------------------------------
    function _synth_case(n)
        rho = collect(range(0.0, 1.0; length=n))
        rmin = 0.6 .* rho
        ne = 8.0 .* (1 .- 0.8 .* rho .^ 2)
        Te = 20.0 .* (1 .- 0.9 .* rho .^ 2) .+ 0.5
        Ti = copy(Te)
        ni = 0.9 .* ne
        volume = 30.0 .* rho .^ 2 .+ 1e-3
        input = ALPHA.AlphaInput{Float64}(; rho, rmin, ne, Te, Ti, ni, volume)
        n_cl, T_eq, _, S0 = ALPHA.slowing_down(ne, Te, Ti, ni; E_alpha=3.5, Z1=5 / 3, ln_lambda=17.0)
        dndr = [0.2 < r < 0.8 ? 0.01 : 0.05 for r in rho]   # small enough for stiff transport to be active
        return input, n_cl, T_eq, S0, dndr
    end
    LEGACY = (; D_interface=false, norm_const=false, plateau_window=0, tol=0.0, warn_nonconverged=false)
    osc_metric(D) = (s = sum(abs.(diff(D))); s > 0 ? sum(abs.(D[3:end] .- 2 .* D[2:end-1] .+ D[1:end-2])) / s : 0.0)

    @testset "legacy flags reproduce the pre-1.1 solver (golden values)" begin
        input, n_cl, T_eq, S0, dndr = _synth_case(31)
        dpdr = dndr .* T_eq
        p = ALPHA.AlphaTransportParams{Float64}(; n_iter=500, i_tot_TAE=0, LEGACY...)
        st = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=dndr, dpdr_crit=dpdr); params=p, critgrad_method=:density)
        @test st.n_iter == 500
        @test st.exit_reason == :max_iter
        @test !st.converged
        @test sum(st.n_tran) ≈ 0.75006382676286665 rtol = 1e-13
        @test st.n_tran[16] ≈ 0.021651712290141949 rtol = 1e-13
        @test maximum(st.D_alpha) ≈ 51.929348638743122 rtol = 1e-13
        @test st.error ≈ 1.7615950216636731 rtol = 1e-13
        @test sum(st.flux) ≈ 0.016220729505483384 rtol = 1e-13
        @test length(st.D_half) == 30
        @test all(2 .* st.D_half .== st.D_alpha[1:end-1] .+ st.D_alpha[2:end])
        p = ALPHA.AlphaTransportParams{Float64}(; n_iter=500, i_tot_TAE=-1, LEGACY...)
        st = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=dndr, dpdr_crit=dpdr); params=p, critgrad_method=:density)
        @test sum(st.n_tran) ≈ 0.74892327413998994 rtol = 1e-13
        @test maximum(st.D_alpha) ≈ 51.626719597373352 rtol = 1e-13
        @test st.error ≈ 1.8022758134174359 rtol = 1e-13
        # QL-coupled run_alpha path (n=21 flat plasma)
        n = 21
        rho = collect(range(0.0, 1.0; length=n))
        inp = ALPHA.AlphaInput{Float64}(; rho, rmin=0.6 .* rho, ne=fill(8.0, n), Te=fill(15.0, n), Ti=fill(15.0, n),
            ni=fill(7.0, n), volume=30.0 .* rho .^ 2, Rmaj=fill(6.2, n))
        tp = ALPHA.AlphaTransportParams{Float64}(; n_iter=300, use_ql_diffusivity=true,
            ql_params=ALPHA.QLDiffusivityParams{Float64}(; km_max=3, dt_update=1e-2), LEGACY...)
        res = run_alpha(inp, (; dndr_crit=fill(0.5, n)); solver=:stiff, transport_params=tp)
        @test sum(res.n_EP) ≈ 0.40524463586310783 rtol = 1e-13
        @test maximum(res.D_alpha) ≈ 32.284886714474901 rtol = 1e-13
        @test res.stiff_error ≈ 0.93996011726022455 rtol = 1e-13
        @test res.stiff_n_iter == 300
        @test !res.stiff_converged
    end

    @testset "single-iteration closure formulas" begin
        input, n_cl, T_eq, S0, dndr = _synth_case(31)
        rho = input.rho; a = input.rmin[end]; n = length(rho)
        dpdr = dndr .* T_eq            # package units 10^19 keV/m -> threshold = dpdr*0.16022 in 10 kPa/m
        base = (; n_iter=1, relax_f=1.0, tol=0.0, plateau_window=0, warn_nonconverged=false, l_crit_smooth=false)
        n0 = n_cl .* (1 .+ 0.01 .* (rho .- 0.5) ./ 0.5); n0[end] = 0.0
        g0 = ALPHA._radial_grad(n0, rho, a)
        p0 = n0 .* T_eq .* 0.16022
        gp0 = ALPHA._radial_grad(p0, rho, a)
        for (norm_const, den_n, den_p) in ((false, n_cl, n_cl .* T_eq .* 0.16022), (true, fill(maximum(n_cl), n), fill(maximum(n_cl .* T_eq .* 0.16022), n)))
            # point closure, density threshold
            st = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=dndr);
                params=ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=0, D_interface=false, norm_const, base...), critgrad_method=:density)
            @test st.D_alpha ≈ 0.001 .+ 7.4 .* max.(g0 .- dndr, 0.0) .* a ./ den_n rtol = 1e-12
            # interface closure, density threshold
            st = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=dndr);
                params=ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=0, D_interface=true, norm_const, base...), critgrad_method=:density)
            gh = [(n0[i] - n0[i+1]) / a / (rho[i+1] - rho[i]) for i in 1:n-1]
            Dh = [0.001 + 7.4 * max(gh[i] - 0.5 * (dndr[i] + dndr[i+1]), 0.0) * a / (0.5 * (den_n[i] + den_n[i+1])) for i in 1:n-1]
            @test st.D_half ≈ Dh rtol = 1e-12
            @test st.D_alpha[1] == st.D_half[1]
            @test st.D_alpha[end] == st.D_half[end]
            @test st.D_alpha[2:end-1] ≈ 0.5 .* (Dh[1:end-1] .+ Dh[2:end]) rtol = 1e-12
            # point + interface closure, pressure threshold (units: dpdr*0.16022)
            st = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dpdr_crit=dpdr);
                params=ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=-1, D_interface=false, norm_const, base...), critgrad_method=:pressure)
            @test st.rg_p_th ≈ dpdr .* 0.16022
            @test st.D_alpha ≈ 0.001 .+ 7.4 .* max.(gp0 .- dpdr .* 0.16022, 0.0) .* a ./ den_p rtol = 1e-12
            st = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dpdr_crit=dpdr);
                params=ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=-1, D_interface=true, norm_const, base...), critgrad_method=:pressure)
            ghp = [(n0[i] * T_eq[i] - n0[i+1] * T_eq[i+1]) * 0.16022 / a / (rho[i+1] - rho[i]) for i in 1:n-1]
            thp = dpdr .* 0.16022
            Dhp = [0.001 + 7.4 * max(ghp[i] - 0.5 * (thp[i] + thp[i+1]), 0.0) * a / (0.5 * (den_p[i] + den_p[i+1])) for i in 1:n-1]
            @test st.D_half ≈ Dhp rtol = 1e-12
        end
    end

    @testset "interface closure removes grid-to-grid oscillation" begin
        input, n_cl, T_eq, S0, dndr = _synth_case(41)
        run(di) = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=dndr);
            params=ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=0, D_interface=di, norm_const=false, n_iter=5000,
                tol=0.0, plateau_window=0, warn_nonconverged=false), critgrad_method=:density)
        sp, si = run(false), run(true)
        @test maximum(si.D_alpha) > 0.01     # stiff transport active
        @test osc_metric(si.D_alpha) < 0.5 * osc_metric(sp.D_alpha)
        signchanges(D) = (d = diff(D); count(i -> d[i] * d[i+1] < 0, 1:length(d)-1))
        @test signchanges(si.D_alpha) < signchanges(sp.D_alpha)
        @test all(isfinite, si.n_tran) && all(>=(0), si.n_tran)
    end

    @testset "convergence: tol, plateau and max_iter exits" begin
        input, n_cl, T_eq, S0, dndr = _synth_case(31)
        big = fill(1e3, 31)   # threshold never reached -> linear problem with D = D_bkg; geometric convergence
        lin = (; i_tot_TAE=0, D_bkg=1.0, tol=0.0, plateau_window=50, plateau_ratio=0.9, n_iter=10_000)
        st = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=big);
            params=ALPHA.AlphaTransportParams{Float64}(; relax=1e-3, lin...), critgrad_method=:density)
        @test st.exit_reason == :plateau
        @test st.converged
        @test st.n_iter == 100          # first window never triggers; second window ratio ≈ 0.999^50 > 0.9
        st2 = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=big);
            params=ALPHA.AlphaTransportParams{Float64}(; relax=0.05, lin...), critgrad_method=:density)
        @test st2.exit_reason == :plateau
        @test st2.n_iter > 100           # 0.95^50 ≈ 0.08 < 0.9: keeps decaying until the round-off floor
        @test st2.n_iter % 50 == 0
        @test st2.error < st.error
        st = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=big);
            params=ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=0, D_bkg=1.0, relax=0.05, tol=1e-6, plateau_window=0, n_iter=10_000), critgrad_method=:density)
        @test st.exit_reason == :tol
        @test st.converged
        @test st.error < 1e-6
        @test_logs (:warn, r"not converged") ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=big);
            params=ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=0, tol=0.0, plateau_window=0, n_iter=20), critgrad_method=:density)
        st = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=big);
            params=ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=0, tol=0.0, plateau_window=0, n_iter=20, warn_nonconverged=false), critgrad_method=:density)
        @test st.n_iter == 20 && st.exit_reason == :max_iter && !st.converged
    end

    @testset "dpdr_crit units and load_crit_grad" begin
        input, n_cl, _, S0, dndr = _synth_case(31)
        n = length(dndr)
        T_flat = fill(500.0, n)
        dpdr = dndr .* T_flat        # 10^19 keV/m; same problem as the density threshold up to a constant
        p = (; n_iter=3000, tol=0.0, plateau_window=0, warn_nonconverged=false, l_crit_smooth=false)
        for nc in (false, true)
            sd = ALPHA.stiff_cgm_transport(input, n_cl, T_flat, S0, (; dndr_crit=dndr);
                params=ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=0, norm_const=nc, p...), critgrad_method=:density)
            sp = ALPHA.stiff_cgm_transport(input, n_cl, T_flat, S0, (; dpdr_crit=dpdr);
                params=ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=-1, norm_const=nc, p...), critgrad_method=:pressure)
            @test maximum(sd.D_alpha) > 0.01
            @test sd.n_tran ≈ sp.n_tran rtol = 1e-6   # rounding amplified by the stiff feedback; a units error is a factor 6 or 39
            @test maximum(abs.(sd.D_alpha .- sp.D_alpha)) < 1e-4 * maximum(sd.D_alpha)   # D is a stiff amplification of the n round-off (observed ~3e-6)
        end
        mktempdir() do dir
            vals = [1.5, 2.5, 1000.0]
            for (f, hdr) in (("alpha_dndr_crit.input", "Density critical gradient (10^19/m^4)"),
                             ("alpha_dpdr_crit.input", "Pressure critical gradient (10 kPa/m)"))
                open(joinpath(dir, f), "w") do io
                    println(io, hdr)
                    foreach(v -> println(io, "  ", v), vals)
                end
            end
            cg = ALPHA.load_crit_grad(; dndr=joinpath(dir, "alpha_dndr_crit.input"), dpdr=joinpath(dir, "alpha_dpdr_crit.input"))
            @test cg.dndr_crit == vals
            @test cg.dpdr_crit ≈ vals ./ 0.16022
            raw = ALPHA.load_crit_grad(; dpdr=joinpath(dir, "alpha_dpdr_crit.input"), convert_units=false)
            @test raw.dpdr_crit == vals
            @test raw.dndr_crit === nothing
        end
    end

    @testset "Vp override and run_alpha parameter forwarding" begin
        input, n_cl, T_eq, S0, dndr = _synth_case(31)
        p = ALPHA.AlphaTransportParams{Float64}(; i_tot_TAE=0, n_iter=200, tol=0.0, plateau_window=0, warn_nonconverged=false)
        s0 = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=dndr); params=p, critgrad_method=:density)
        s1 = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=dndr); params=p, critgrad_method=:density,
            Vp=ALPHA._vprime(input.volume, input.rmin))
        @test s1.n_tran == s0.n_tran
        Vp_f = ALPHA.vprime_fortran(fill(1.5, 31), input.rmin, fill(1.7, 31))
        s2 = ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=dndr); params=p, critgrad_method=:density, Vp=Vp_f)
        @test !(s2.n_tran ≈ s0.n_tran)
        @test_throws ArgumentError ALPHA.stiff_cgm_transport(input, n_cl, T_eq, S0, (; dndr_crit=dndr); params=p, Vp=ones(5))
        # run_alpha must forward every AlphaTransportParams field (only i_tot_TAE is overridden)
        tp = ALPHA.AlphaTransportParams{Float64}(; D_interface=false, norm_const=false, n_iter=150, tol=0.0, plateau_window=0,
            warn_nonconverged=false, relax=2e-3, D_TAE=3.3)
        res = run_alpha(input, (; dndr_crit=dndr); solver=:stiff, method=:density, transport_params=tp)
        nc, Te, _, S = ALPHA.slowing_down(input.ne, input.Te, input.Ti, input.ni; E_alpha=3.5, Z1=5 / 3, ln_lambda=17.0)
        direct = ALPHA.stiff_cgm_transport(input, nc, Te, S, (; dndr_crit=dndr); params=ALPHA._with(tp; i_tot_TAE=0), critgrad_method=:density)
        @test res.n_EP == direct.n_tran
        @test res.stiff_n_iter == 150
        @test res.stiff_exit_reason == :max_iter
        @test !res.stiff_converged
    end

    @testset "dual EP species keeps the point closure" begin
        n = 21
        rho = collect(range(0.0, 1.0; length=n))
        rmin = 0.6 .* rho
        input = ALPHA.AlphaInput{Float64}(; rho, rmin, Rmaj=6.2 .- 0.5 .* rmin,
            ne=8.0 .* (1 .- 0.5 .* rho), Te=20.0 .* (1 .- 0.8 .* rho) .+ 1.0,
            Ti=fill(15.0, n), ni=fill(7.0, n), volume=40.0 .* rho .^ 2)
        dndr = fill(0.01, n)
        run(di) = run_alpha(input, (; dndr_crit=dndr, dndr_crit2=dndr); solver=:stiff, ep_mode=:fusion_nbi, method=:density,
            transport_params=ALPHA.AlphaTransportParams{Float64}(; n_iter=300, tol=0.0, plateau_window=0, warn_nonconverged=false, D_interface=di))
        r0, r1 = run(false), run(true)
        @test length(r1.n_EP2) == n && all(isfinite, r1.n_EP2)
        @test r1.n_EP2 == r0.n_EP2        # species 2 is untouched by the interface closure of species 1
    end

    @testset "Fortran Alpha (jlestz) DIII-D 153071 parity" begin
        reldev(x, ref; floor=1e-3 * maximum(abs.(ref))) = maximum(abs.(x .- ref) ./ max.(abs.(ref), floor))
        common = (; D_bkg=0.001, D_TAE=7.4, relax=5e-4, relax_f=0.005, delta0=0.01, delta1=0.0, rdelta0=0.5,
            SDsink=1.0, i_tot_TAE=-1, plateau_window=0, warn_nonconverged=false)
        for (t, spike) in (("3400", :axis), ("2200", :edge))
            fx = ALPHA.read_alpha_fixture(joinpath(@__DIR__, "fixtures", "d3d_153071_$(t)_alpha.txt"))
            n = length(fx.rho)
            @test n == 101
            input = ALPHA.AlphaInput{Float64}(; rho=fx.rho, rmin=fx.rmin, Rmaj=fx.Rmaj,
                ne=ones(n), Te=ones(n), Ti=ones(n), ni=ones(n), volume=zeros(n))
            dndr = min.(fx.dndr, 1000.0)
            dpdr = min.(fx.dpdr, 1000.0) ./ 0.16022
            axis = fx.rho .< 0.05; edge = fx.rho .> 0.9
            # old scheme: dndr-derived pressure threshold, point closure, local normalisation, exactly 10000 iterations
            po = ALPHA.AlphaTransportParams{Float64}(; D_interface=false, norm_const=false, n_iter=fx.meta[:old_n_iter], tol=0.0, common...)
            so = ALPHA.stiff_cgm_transport(input, fx.n_cl, fx.T_equiv, fx.S0, (; dndr_crit=dndr); params=po, critgrad_method=:density, Vp=fx.Vp)
            @test so.n_iter == fx.meta[:old_n_iter] && so.exit_reason == :max_iter
            @test reldev(so.rg_p_th, fx.rg_p_th_old) < 1e-6
            @test reldev(so.n_tran, fx.n_old) < 1e-6
            @test maximum(abs.(so.D_alpha .- fx.D_old) ./ max.(fx.D_old, 1e-3)) < 1e-5
            @test reldev(so.flux .* fx.Vp, fx.fluxVp_old) < 1e-6
            @test so.error ≈ fx.meta[:old_error_final] rtol = 1e-4
            s1 = ALPHA.stiff_cgm_transport(input, fx.n_cl, fx.T_equiv, fx.S0, (; dndr_crit=dndr);
                params=ALPHA.AlphaTransportParams{Float64}(; D_interface=false, norm_const=false, n_iter=1, tol=0.0, common...), critgrad_method=:density, Vp=fx.Vp)
            @test s1.error ≈ fx.meta[:old_errlog][1] rtol = 1e-9
            # new scheme: dpdr threshold, interface closure, max normalisation, tol=1e-12
            pn = ALPHA.AlphaTransportParams{Float64}(; D_interface=true, norm_const=true, n_iter=100_000, tol=1e-12, common...)
            sn = ALPHA.stiff_cgm_transport(input, fx.n_cl, fx.T_equiv, fx.S0, (; dpdr_crit=dpdr); params=pn, critgrad_method=:pressure, Vp=fx.Vp)
            @test sn.exit_reason == :tol && sn.converged && sn.error < 1e-12
            @test abs(sn.n_iter - fx.meta[:new_n_iter]) <= 0.005 * fx.meta[:new_n_iter]
            @test reldev(sn.rg_p_th, fx.rg_p_th_new) < 1e-6
            @test reldev(sn.n_tran, fx.n_new) < 1e-6
            @test maximum(abs.(sn.D_alpha .- fx.D_new) ./ max.(fx.D_new, 1e-3)) < 1e-5
            @test reldev(sn.flux .* fx.Vp, fx.fluxVp_new) < 1e-6
            s1 = ALPHA.stiff_cgm_transport(input, fx.n_cl, fx.T_equiv, fx.S0, (; dpdr_crit=dpdr);
                params=ALPHA.AlphaTransportParams{Float64}(; D_interface=true, norm_const=true, n_iter=1, tol=0.0, common...), critgrad_method=:pressure, Vp=fx.Vp)
            @test s1.error ≈ fx.meta[:new_errlog][1] rtol = 1e-9
            # physics: the spikes of the old scheme are gone in the new one
            @test maximum(sn.D_alpha[axis]) < 1.0 && maximum(sn.D_alpha[edge]) < 1.0
            spike === :axis && @test maximum(so.D_alpha[axis]) > 100.0
            spike === :edge && @test maximum(so.D_alpha[edge]) > 5.0
            # attribution: the interface closure removes the zig-zag (dpdr threshold, max normalisation;
            # the edge spike of the local normalisation would otherwise dominate the metric at t=2200)
            sp_pt = ALPHA.stiff_cgm_transport(input, fx.n_cl, fx.T_equiv, fx.S0, (; dpdr_crit=dpdr);
                params=ALPHA.AlphaTransportParams{Float64}(; D_interface=false, norm_const=true, n_iter=100_000, tol=1e-12, common...), critgrad_method=:pressure, Vp=fx.Vp)
            @test osc_metric(sn.D_alpha) < 0.5 * osc_metric(sp_pt.D_alpha)
            # ... and the max normalisation removes the edge spike (point closure, dpdr threshold)
            sp_loc = ALPHA.stiff_cgm_transport(input, fx.n_cl, fx.T_equiv, fx.S0, (; dpdr_crit=dpdr);
                params=ALPHA.AlphaTransportParams{Float64}(; D_interface=false, norm_const=false, n_iter=100_000, tol=1e-12, common...), critgrad_method=:pressure, Vp=fx.Vp)
            spike === :edge && @test maximum(sp_loc.D_alpha[edge]) > 5.0 * maximum(sp_pt.D_alpha[edge])
        end
    end

    @testset "nbi + He ash" begin
        n = 21
        rho = range(0.0, 1.0; length=n) |> collect
        rmin = 0.6 .* rho
        Rmaj = 6.2 .- 0.5 .* rmin
        input = ALPHA.AlphaInput{Float64}(;
            rho, rmin, Rmaj,
            ne=8.0 .* (1 .- 0.5 .* rho), Te=20.0 .* (1 .- 0.8 .* rho) .+ 1.0,
            Ti=fill(15.0, n), ni=fill(7.0, n), volume=40.0 .* rho .^ 2)
        S_nbi = ALPHA.nbi_pencil_beam_source(input; nbi=ALPHA.NBIBeamParams{Float64}(; Pow_NBI=10.0))
        @test all(S_nbi .>= 0)
        @test any(S_nbi .> 0)
        dndr = fill(1.0, n)
        tp = ALPHA.AlphaTransportParams{Float64}(;
            n_iter=200, tol=0.05, relax=1e-3, warn_nonconverged=false,
            he_ash_params=ALPHA.HeAshParams{Float64}(; n_iter=100, tol=0.05))
        res = run_alpha(input, (; dndr_crit=dndr, dndr_crit2=dndr); solver=:stiff,
            ep_mode=:fusion_nbi, transport_params=tp)
        @test length(res.n_EP2) == n
        @test any(res.n_EP2 .> 0)
        @test length(res.n_He) == n
        @test all(res.n_He .>= 0)
    end

end
