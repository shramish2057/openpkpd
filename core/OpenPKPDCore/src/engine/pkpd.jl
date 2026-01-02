export simulate_pkpd

"""
simulate_pkpd runs PK first, then evaluates PD from a named PK observation series.

Rules:
- PD input must exist in pk_result.observations
- Output merges observations, does not overwrite keys silently
"""
function simulate_pkpd(
    pk_spec::ModelSpec, pd_spec::PDSpec, grid::SimGrid, solver::SolverSpec
)
    pk_res = simulate(pk_spec, grid, solver)

    inkey = pd_spec.input_observation
    outkey = pd_spec.output_observation

    if !haskey(pk_res.observations, inkey)
        error("PD input observation $(inkey) not found in PK observations")
    end

    if haskey(pk_res.observations, outkey)
        error("PD output observation $(outkey) already exists in PK observations")
    end

    input_series = pk_res.observations[inkey]
    pd_series = evaluate(pd_spec, input_series)

    observations = copy(pk_res.observations)
    observations[outkey] = pd_series

    metadata = copy(pk_res.metadata)
    metadata["pd_model"] = string(typeof(pd_spec.kind))
    metadata["pd_name"] = pd_spec.name
    metadata["pd_input_observation"] = String(inkey)
    metadata["pd_output_observation"] = String(outkey)
    metadata["pd_params"] = Dict(
        "E0" => pd_spec.params.E0,
        "Emax" => pd_spec.params.Emax,
        "EC50" => pd_spec.params.EC50,
    )

    return SimResult(pk_res.t, pk_res.states, observations, metadata)
end
