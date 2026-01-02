using Test
using OpenPKPDCore

function analytic_onecomp_ivbolus_conc(
    t::Float64, dose::Float64, CL::Float64, V::Float64, t_dose::Float64
)
    if t < t_dose
        return 0.0
    end
    k = CL / V
    return (dose / V) * exp(-k * (t - t_dose))
end

@testset "OneCompIVBolus analytic equivalence" begin
    spec = ModelSpec(
        OneCompIVBolus(), "1c_iv_bolus", Dict(:CL => 5.0, :V => 50.0), 100.0, 0.0
    )

    grid = SimGrid(0.0, 24.0, collect(0.0:0.5:24.0))

    solver = SolverSpec(:Tsit5, 1e-10, 1e-12, 10^7)

    res = simulate(spec, grid, solver)

    @test length(res.t) == length(grid.saveat)
    @test res.t == grid.saveat

    CL = spec.params[:CL]
    V = spec.params[:V]

    for (i, t) in enumerate(res.t)
        c_ref = analytic_onecomp_ivbolus_conc(t, spec.dose_amount, CL, V, spec.dose_time)
        @test isapprox(res.conc[i], c_ref; rtol=1e-8, atol=1e-10)
    end
end

@testset "Dose event at nonzero time" begin
    spec = ModelSpec(
        OneCompIVBolus(), "1c_iv_bolus_delayed", Dict(:CL => 3.0, :V => 30.0), 120.0, 2.0
    )

    grid = SimGrid(0.0, 12.0, collect(0.0:0.25:12.0))
    solver = SolverSpec(:Tsit5, 1e-10, 1e-12, 10^7)

    res = simulate(spec, grid, solver)

    CL = spec.params[:CL]
    V = spec.params[:V]

    for (i, t) in enumerate(res.t)
        if t == spec.dose_time
            continue  # skip discontinuity
        end
        c_ref = analytic_onecomp_ivbolus_conc(t, spec.dose_amount, CL, V, spec.dose_time)
        @test isapprox(res.conc[i], c_ref; rtol=1e-8, atol=1e-10)
    end
end
