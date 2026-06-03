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
        res = run_alpha(input, (; dndr_crit); method=:density)

        @test length(res.n_EP) == n
        @test all(res.n_EP .>= 0)
        @test all(res.n_EP .<= res.n_classical .+ 1e-9)   # transport can only flatten
        @test all(isfinite, res.flux_particle)
        @test all(isfinite, res.flux_energy)
        @test all(res.T_EP .≈ res.T_alpha_equiv)          # :density => T from slowing-down
        @test any(res.transport_active)                    # some region is AE-limited
        # pressure method
        dpdr_crit = dndr_crit .* res.T_alpha_equiv
        res2 = run_alpha(input, (; dndr_crit, dpdr_crit); method=:pressure)
        @test all(res2.p_EP .>= 0)
        @test all(isfinite, res2.T_EP)
    end

end
