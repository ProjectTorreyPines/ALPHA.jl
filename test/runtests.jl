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
        params = ALPHA.AlphaTransportParams{Float64}(; n_iter=500, tol=1e-2, relax=1e-3)
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
            n_iter=300, tol=1e-2, relax=1e-3, use_ql_diffusivity=true, ql_params=params)
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
            n_iter=200, tol=0.05, relax=1e-3,
            he_ash_params=ALPHA.HeAshParams{Float64}(; n_iter=100, tol=0.05))
        res = run_alpha(input, (; dndr_crit=dndr, dndr_crit2=dndr); solver=:stiff,
            ep_mode=:fusion_nbi, transport_params=tp)
        @test length(res.n_EP2) == n
        @test any(res.n_EP2 .> 0)
        @test length(res.n_He) == n
        @test all(res.n_He .>= 0)
    end

end
