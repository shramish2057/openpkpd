module OpenPKPDCore

using SciMLBase
using DifferentialEquations

export ModelKind, OneCompIVBolus, ModelSpec, SolverSpec, SimGrid, SimResult, simulate

# -------------------------
# Model definitions
# -------------------------

abstract type ModelKind end

"""
One-compartment IV bolus PK model.

State:
- A(t): amount in central compartment

Parameters:
- CL: clearance
- V: central volume

Dynamics:
dA/dt = -(CL/V) * A

Observation:
C(t) = A(t) / V
"""
struct OneCompIVBolus <: ModelKind end

"""
Generic model specification container.

Rules:
- spec is pure data
- no solver configuration inside model spec
"""
struct ModelSpec{K<:ModelKind}
    kind::K
    name::String
    params::Dict{Symbol,Float64}
    dose_amount::Float64
    dose_time::Float64
end

"""
Solver specification is pure data.
No hidden defaults in the engine.
"""
struct SolverSpec
    alg::Symbol
    reltol::Float64
    abstol::Float64
    maxiters::Int
end

"""
Simulation time grid definition.
We require explicit save times for deterministic output shapes.
"""
struct SimGrid
    t0::Float64
    t1::Float64
    saveat::Vector{Float64}
end

"""
Simulation result.
Contains deterministic outputs and metadata.
"""
struct SimResult
    t::Vector{Float64}
    amount::Vector{Float64}
    conc::Vector{Float64}
    metadata::Dict{String,Any}
end

# -------------------------
# Validation helpers
# -------------------------

function _require(keys::Vector{Symbol}, params::Dict{Symbol,Float64})
    missing = Symbol[]
    for k in keys
        if !haskey(params, k)
            push!(missing, k)
        end
    end
    if !isempty(missing)
        error("Missing required parameters: $(missing)")
    end
    return nothing
end

function _require_positive(name::String, x::Float64)
    if !(x > 0.0)
        error("Expected positive value for $(name), got $(x)")
    end
    return nothing
end

function validate(spec::ModelSpec{OneCompIVBolus})
    _require([:CL, :V], spec.params)

    CL = spec.params[:CL]
    V = spec.params[:V]

    _require_positive("CL", CL)
    _require_positive("V", V)
    _require_positive("dose_amount", spec.dose_amount)

    if spec.dose_time < 0.0
        error("dose_time must be >= 0, got $(spec.dose_time)")
    end

    return nothing
end

function validate(grid::SimGrid)
    _require_positive("t1", grid.t1)
    if grid.t0 < 0.0
        error("t0 must be >= 0, got $(grid.t0)")
    end
    if !(grid.t1 > grid.t0)
        error("t1 must be > t0")
    end
    if isempty(grid.saveat)
        error("saveat must not be empty")
    end
    if any(t -> t < grid.t0 || t > grid.t1, grid.saveat)
        error("All saveat values must be within [t0, t1]")
    end
    if !issorted(grid.saveat)
        error("saveat must be sorted ascending")
    end
    return nothing
end

function validate(solver::SolverSpec)
    _require_positive("reltol", solver.reltol)
    _require_positive("abstol", solver.abstol)
    if solver.maxiters < 1
        error("maxiters must be >= 1")
    end
    return nothing
end

# -------------------------
# Solver mapping
# -------------------------

function _solver_alg(alg::Symbol)
    if alg == :Tsit5
        return Tsit5()
    elseif alg == :Rosenbrock23
        return Rosenbrock23()
    else
        error("Unsupported solver alg: $(alg). Supported: :Tsit5, :Rosenbrock23")
    end
end

# -------------------------
# ODE definitions
# -------------------------

function _ode_onecomp_ivbolus!(dA, A, p, t)
    CL = p.CL
    V = p.V
    dA[1] = -(CL / V) * A[1]
    return nothing
end

# -------------------------
# Main simulation entrypoint
# -------------------------

function simulate(spec::ModelSpec{OneCompIVBolus}, grid::SimGrid, solver::SolverSpec)
    validate(spec)
    validate(grid)
    validate(solver)

    CL = spec.params[:CL]
    V = spec.params[:V]

    # Dose handling for IV bolus:
    # For v1 we support one bolus at dose_time.
    # Implemented as: initial condition at t0 plus event if dose_time > t0.
    A0 = 0.0
    if spec.dose_time == grid.t0
        A0 += spec.dose_amount
    end

    p = (CL=CL, V=V)

    u0 = [A0]
    tspan = (grid.t0, grid.t1)

    prob = ODEProblem(_ode_onecomp_ivbolus!, u0, tspan, p)

    # Add dose event if dose_time within (t0, t1]
    cb = nothing
    if spec.dose_time > grid.t0 && spec.dose_time <= grid.t1
        condition(u, t, integrator) = t - spec.dose_time
        function affect!(integrator)
            integrator.u[1] += spec.dose_amount
        end
        cb = ContinuousCallback(condition, affect!)
    end

    sol = solve(
        prob,
        _solver_alg(solver.alg);
        reltol=solver.reltol,
        abstol=solver.abstol,
        maxiters=solver.maxiters,
        saveat=grid.saveat,
        callback=cb,
    )

    A = [u[1] for u in sol.u]
    C = [a / V for a in A]

    metadata = Dict{String,Any}(
        "engine_version" => "0.1.0",
        "model" => "OneCompIVBolus",
        "solver_alg" => String(solver.alg),
        "reltol" => solver.reltol,
        "abstol" => solver.abstol,
        "dose_amount" => spec.dose_amount,
        "dose_time" => spec.dose_time,
        "deterministic_output_grid" => true,
    )

    return SimResult(Vector{Float64}(sol.t), A, C, metadata)
end

end
