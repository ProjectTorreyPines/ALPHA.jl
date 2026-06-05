# Port of Alpha_transport.f90 He-ash transport (i_He_tran=1).

"""Parameters for helium-ash transport."""
Base.@kwdef struct HeAshParams{T<:Real}
    enabled::Bool = true
    edge_fraction::T = T(0.05)     # n_He(edge) = edge_fraction * (ne - ne_edge)
    C_p_He::T = T(-2.0)            # Angioni pinch coefficient
    n_iter::Int = 500
    tol::T = 1.0e-3
    relax::T = 5.0e-4
end

"""Result of He ash transport solve."""
Base.@kwdef struct HeAshResult{T<:Real}
    n_He::Vector{T}
    D_He::Vector{T}
    error::T
    n_iter::Int
end

"""
    he_ash_transport(input, n_alpha_tran, n_alpha_cl, S_alpha; kwargs...) -> HeAshResult

Solve helium ash density with Angioni ITG/TEM diffusivity tied to alpha energy flux
(Fortran `i_He_tran=1` block): uses the alpha particle flux from the fusion source.
"""
function he_ash_transport(
    input::AlphaInput{T},
    n_alpha_tran::AbstractVector{T},
    n_alpha_cl::AbstractVector{T},
    S_alpha::AbstractVector{T};
    params::HeAshParams{T}=HeAshParams{T}(),
    Rmaj::AbstractVector{T}=input.Rmaj,
    chi_eff::Union{AbstractVector{T},Nothing}=nothing,
) where {T<:Real}
    params.enabled || return HeAshResult{T}(; n_He=zeros(T, length(input.rho)), D_He=zeros(T, length(input.rho)),
        error=zero(T), n_iter=0)

    rho, rmin = input.rho, input.rmin
    n = length(rho)
    a = rmin[end]
    dr = _dr_grid(rmin)
    Vp = _vprime(input.volume, rmin)
    ne, Te, Ti, ni = input.ne, input.Te, input.Ti, input.ni

    n_He = [(ne[i] - ne[n]) * params.edge_fraction for i in 1:n]
    n_He[n] = zero(T)

    if chi_eff === nothing
        flux_src = _source_flux(S_alpha, Vp, dr, n_alpha_tran, n_alpha_cl, one(T))
        chi_eff = _angioni_D_bkg(flux_src, Vp, ne, Te, Ti, ni, dr, input.E_alpha, T(10.0))
    end
    D_bkg_He = copy(chi_eff)

    flux = _he_alpha_flux(S_alpha, Vp, dr, n_alpha_tran, n_alpha_cl)
    error = T(Inf)
    n_done = 0
    D_He = similar(n_He)

    for iter in 1:params.n_iter
        n_prev = copy(n_He)
        rg_n = _radial_grad(n_He, rho, a)
        for i in 1:n
            D_He[i] = D_bkg_He[i]
            if i != 1 && i != n && abs(rg_n[i]) > eps(T)
                D_He[i] *= (1 + params.C_p_He / Rmaj[i] * n_He[i] / rg_n[i])
            end
        end
        D_He[1] = D_He[2]
        D_He[n] = D_He[n-1]

        for i in n-1:-1:1
            Dsum = max(D_He[i] + D_He[i+1], eps(T))
            n_He[i] = n_He[i+1] + dr[i] * (flux[i] + flux[i+1]) / Dsum
            n_He[i] = max(n_He[i], zero(T))
        end

        err = zero(T)
        for i in 2:n-1
            e = (n_He[i] - n_prev[i]) / max(abs(n_prev[i]), eps(T))
            err += e^2
        end
        error = sqrt(err / max(n - 2, 1))
        n_done = iter
        n_He .= params.relax .* n_He .+ (1 - params.relax) .* n_prev
        error < params.tol && break
    end

    return HeAshResult{T}(; n_He, D_He, error, n_iter=n_done)
end

"""Alpha-driven He source flux (renormalized ash source, Fortran He block)."""
function _he_alpha_flux(S0, Vp, dr, n_tran, n_cl)
    T = eltype(S0)
    n = length(S0)
    flux = zeros(T, n)
    for i in 2:n
        flux[i] = Vp[i-1] / max(Vp[i], eps(T)) * flux[i-1] +
                  T(0.5) * dr[i] / max(Vp[i], eps(T)) *
                  (Vp[i] * S0[i] * n_tran[i] / max(n_cl[i], eps(T)) +
                   Vp[i-1] * S0[i-1] * n_tran[i-1] / max(n_cl[i-1], eps(T)))
    end
    return flux
end
