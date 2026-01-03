<p align="center">
  <h1 align="center">OpenPKPD</h1>
  <p align="center">
    <strong>Transparent, validated pharmacokinetics and pharmacodynamics modeling infrastructure</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#documentation">Documentation</a> •
    <a href="#contributing">Contributing</a>
  </p>
</p>

---

## Overview

**OpenPKPD** is a reference-grade PK/PD simulation platform written in Julia, designed for research, method development, and reproducible scientific computation. It emphasizes deterministic execution, transparent numerical semantics, and complete artifact serialization.

> **Note:** This is a scientific computing library, not a GUI product or regulatory submission tool.

## Features

### PK Models
- **One-Compartment IV Bolus** — Single compartment with intravenous bolus injection
- **One-Compartment Oral First-Order** — Single compartment with first-order absorption

### PD Models
- **Direct Emax** — Sigmoidal effect model: `Effect(C) = E0 + (Emax × C) / (EC50 + C)`
- **Indirect Response Turnover** — Inhibition of Kout with turnover kinetics

### Population Simulation
- **Inter-Individual Variability (IIV)** — Log-normal distributions with seeded RNG
- **Inter-Occasion Variability (IOV)** — Occasion-based parameter shifts
- **Covariate Effects** — Linear, power, and exponential transformation models
- **Population Summaries** — Mean, median, and quantile statistics

### Sensitivity Analysis
- Single-run and population-level parameter perturbation
- Relative perturbation plans with comprehensive metrics

### Deterministic Execution
- Versioned numerical semantics (event handling, solver behavior)
- Complete artifact serialization and replay
- Reproducible results with StableRNG

## Installation

### Prerequisites

- [Julia](https://julialang.org/downloads/) 1.9 or later

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/openpkpd.git
cd openpkpd

# Navigate to the core package
cd core/OpenPKPDCore

# Activate and instantiate the environment
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Verify Installation

```bash
julia --project=. -e 'using OpenPKPDCore; println("OpenPKPD loaded successfully!")'
```

## Quick Start

### Basic PK Simulation

```julia
using OpenPKPDCore

# Define a one-compartment IV bolus model
pk = ModelSpec(
    OneCompIVBolus(),
    "my_pk_model",
    OneCompIVBolusParams(CL=5.0, V=50.0),  # Clearance, Volume
    [DoseEvent(0.0, 100.0)]                 # Dose at t=0, amount=100
)

# Define simulation grid
grid = SimGrid(0.0, 24.0, collect(0.0:0.5:24.0))

# Configure solver with explicit tolerances
solver = SolverSpec(:Tsit5, 1e-10, 1e-12, 10^7)

# Run simulation
result = simulate(pk, grid, solver)

# Access results
println("Time points: ", result.t)
println("Concentrations: ", result.observations[:conc])
```

### PKPD Simulation

```julia
using OpenPKPDCore

# PK model
pk = ModelSpec(
    OneCompIVBolus(),
    "pk_iv",
    OneCompIVBolusParams(5.0, 50.0),
    [DoseEvent(0.0, 100.0)]
)

# PD model (Direct Emax)
pd = PDSpec(
    DirectEmax(),
    "pd_emax",
    DirectEmaxParams(E0=10.0, Emax=40.0, EC50=0.8),
    :conc,      # Input observation from PK
    :effect     # Output observation name
)

grid = SimGrid(0.0, 24.0, collect(0.0:0.5:24.0))
solver = SolverSpec(:Tsit5, 1e-10, 1e-12, 10^7)

# Run sequential PKPD
result = simulate_pkpd(pk, pd, grid, solver)

println("Effect: ", result.observations[:effect])
```

### Population Simulation with IIV

```julia
using OpenPKPDCore

# Base model
base = ModelSpec(
    OneCompIVBolus(),
    "pop_pk",
    OneCompIVBolusParams(5.0, 50.0),
    [DoseEvent(0.0, 100.0)]
)

# IIV specification (20 individuals, log-normal variability)
iiv = IIVSpec(
    LogNormalIIV(),
    Dict(:CL => 0.2, :V => 0.1),  # Omega values
    UInt64(12345),                 # Seed for reproducibility
    20                             # Number of individuals
)

pop = PopulationSpec(base, iiv, nothing, nothing, IndividualCovariates[])

grid = SimGrid(0.0, 24.0, collect(0.0:1.0:24.0))
solver = SolverSpec(:Tsit5, 1e-9, 1e-11, 10^7)

# Run population simulation
result = simulate_population(pop, grid, solver)

# Access summaries
summary = result.summaries[:conc]
println("Mean concentration: ", summary.mean)
println("Median concentration: ", summary.median)
```

### Sensitivity Analysis

```julia
using OpenPKPDCore

spec = ModelSpec(
    OneCompIVBolus(),
    "sens_pk",
    OneCompIVBolusParams(5.0, 50.0),
    [DoseEvent(0.0, 100.0)]
)

grid = SimGrid(0.0, 24.0, collect(0.0:1.0:24.0))
solver = SolverSpec(:Tsit5, 1e-10, 1e-12, 10^7)

# Define perturbation plan (10% increase in CL)
plan = PerturbationPlan(
    "CL_up_10pct",
    [Perturbation(RelativePerturbation(), :CL, 0.10)]
)

# Run sensitivity analysis
result = run_sensitivity(spec, grid, solver; plan=plan, observation=:conc)

println("Max absolute delta: ", result.metrics.max_abs_delta)
```

## Project Structure

```
openpkpd/
├── core/                          # Julia core engine
│   └── OpenPKPDCore/
│       ├── src/
│       │   ├── specs/             # Data specifications
│       │   ├── models/            # PK model definitions
│       │   ├── pd/                # PD model definitions
│       │   ├── engine/            # Simulation engine
│       │   └── serialization/     # Artifact persistence
│       └── test/                  # Test suite
├── interfaces/                    # CLI and Python bindings
│   ├── cli/
│   └── python/
├── validation/                    # Reference models & golden artifacts
│   ├── golden/                    # Validated reference outputs
│   └── scripts/                   # Validation runners
├── benchmarks/                    # Performance benchmarks
├── docs/                          # Technical documentation
└── examples/                      # Reproducible examples
```

## Running Tests

```bash
cd core/OpenPKPDCore
julia --project=. -e 'include("test/runtests.jl")'
```

## Design Principles

### Non-Negotiable Invariants

- **Deterministic Execution** — Same inputs always produce identical outputs
- **Explicit Configuration** — No hidden defaults; all solver parameters must be specified
- **Versioned Semantics** — Breaking numerical changes require explicit version bumps
- **Complete Serialization** — All model specifications and results are persistable

### Semantic Versioning

| Version | Description |
|---------|-------------|
| `EVENT_SEMANTICS_VERSION` | Dose event handling behavior |
| `SOLVER_SEMANTICS_VERSION` | ODE solver algorithm mapping |
| `ARTIFACT_SCHEMA_VERSION` | Serialization data schema |

## Documentation

| Document | Description |
|----------|-------------|
| [DESIGN.md](./DESIGN.md) | Architectural design principles |
| [EVENT_SEMANTICS.md](./EVENT_SEMANTICS.md) | Dose event semantics v1.0.0 |
| [SOLVER_SEMANTICS.md](./SOLVER_SEMANTICS.md) | Solver semantics v1.0.0 |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | Contribution guidelines |
| [validation/CHANGE_POLICY.md](./validation/CHANGE_POLICY.md) | Change policy and versioning |

## Supported Solvers

| Solver | Description |
|--------|-------------|
| `:Tsit5` | 5th order Runge-Kutta (Tsitouras) |
| `:Rosenbrock23` | Linearly implicit Rosenbrock method |

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments

- Built with [DifferentialEquations.jl](https://diffeq.sciml.ai/) for ODE solving
- Uses [StableRNGs.jl](https://github.com/JuliaRandom/StableRNGs.jl) for reproducible random number generation

---

<p align="center">
  <sub>Built for reproducible pharmacometric research</sub>
</p>
