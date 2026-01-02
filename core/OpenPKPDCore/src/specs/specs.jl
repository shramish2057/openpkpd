# Specifications are pure data. No solver logic and no hidden defaults.
# -------------------------
# Shared validation helpers
# -------------------------

function _require_positive(name::String, x::Float64)
    if !(x > 0.0)
        error("Expected positive value for $(name), got $(x)")
    end
    return nothing
end

# -------------------------
# PK specifications
# -------------------------
export ModelKind,
    OneCompIVBolus, OneCompOralFirstOrder, OneCompIVBolusParams, OneCompOralFirstOrderParams
export DoseEvent, ModelSpec
export SolverSpec, SimGrid, SimResult

abstract type ModelKind end

struct OneCompIVBolus <: ModelKind end
struct OneCompOralFirstOrder <: ModelKind end

struct DoseEvent
    time::Float64
    amount::Float64
end

struct OneCompIVBolusParams
    CL::Float64
    V::Float64
end

struct OneCompOralFirstOrderParams
    Ka::Float64
    CL::Float64
    V::Float64
end

struct ModelSpec{K<:ModelKind,P}
    kind::K
    name::String
    params::P
    doses::Vector{DoseEvent}
end

struct SolverSpec
    alg::Symbol
    reltol::Float64
    abstol::Float64
    maxiters::Int
end

struct SimGrid
    t0::Float64
    t1::Float64
    saveat::Vector{Float64}
end

struct SimResult
    t::Vector{Float64}
    states::Dict{Symbol,Vector{Float64}}
    observations::Dict{Symbol,Vector{Float64}}
    metadata::Dict{String,Any}
end

# -------------------------
# PD specifications
# -------------------------

export PDModelKind, DirectEmax, DirectEmaxParams, PDSpec

abstract type PDModelKind end

"""
Direct Emax PD model.

Effect(C) = E0 + (Emax * C) / (EC50 + C)
"""
struct DirectEmax <: PDModelKind end

struct DirectEmaxParams
    E0::Float64
    Emax::Float64
    EC50::Float64
end

"""
PD specification container.

input_observation:
- which observation key from the upstream system is used as input, usually :conc

output_observation:
- name of the produced PD observable, default :effect is typical
"""
struct PDSpec{K<:PDModelKind,P}
    kind::K
    name::String
    params::P
    input_observation::Symbol
    output_observation::Symbol
end
