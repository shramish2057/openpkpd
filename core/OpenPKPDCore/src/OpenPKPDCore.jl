module OpenPKPDCore

using SciMLBase
using DifferentialEquations

include("specs/specs.jl")
include("models/onecomp_iv_bolus.jl")
include("engine/solve.jl")

end
