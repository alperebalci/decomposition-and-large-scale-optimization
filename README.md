# Decomposition and Large-Scale Optimization

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary Jors Academy methodology umbrella for decomposition algorithms and scalable optimization. It complements the focused [Benders Decomposition Methods](https://github.com/jorsacademy/benders-decomposition-methods) repository by covering a broader family of decomposition architectures and by providing small exact verification instances.

<!-- portfolio-umbrella:end -->

The repository focuses on one question:

> How can a large optimization model be separated into smaller structured problems while preserving useful bounds, feasibility guarantees, and reproducible convergence diagnostics?

## Native research modules

| Module | Decomposition idea | Benchmark |
|---|---|---|
| Benders decomposition | Separate binary design decisions from continuous recourse | Capacitated facility location |
| Column generation | Solve a restricted master and generate improving columns by pricing | Cutting stock |\n| Branch-and-price | Combine column generation with Ryan-Foster branching that is enforced inside pricing | Bin packing / set partitioning |
| Lagrangian relaxation | Dualize a complicating resource constraint and optimize separable subproblems | 0-1 knapsack relaxation |
| Progressive Hedging | Enforce nonanticipativity across scenario subproblems with augmented penalties | Two-stage stochastic production |
| Column-and-constraint generation | Alternate a decision master with an adversarial uncertainty subproblem | Budgeted robust production |\n| Consensus ADMM | Split a shared decision into agent-local copies and enforce agreement with augmented-Lagrangian updates | Distributed strongly convex quadratic optimization |

The examples are intentionally small enough to verify independently, but the implementations expose the algorithmic objects that matter at scale: master bounds, subproblem duals, reduced costs, multiplier updates, incumbent recovery, and convergence gaps.

## Method map

```text
Large structured optimization
├── Primal decomposition
│   ├── Benders decomposition
│   └── Logic-based Benders
├── Dual / price decomposition
│   ├── Lagrangian relaxation
│   ├── Dantzig-Wolfe decomposition
│   ├── Column generation\n│   └── Branch-and-price
├── Scenario decomposition
│   ├── Progressive Hedging
│   └── L-shaped methods
├── Consensus / augmented-Lagrangian decomposition
│   └── ADMM
└── Robust / adversarial decomposition
    └── Column-and-constraint generation
```

## Installation

```bash
python -m pip install -e ".[dev]"
pytest
```

Python 3.10+ is supported. The reference implementations use NumPy and SciPy/HiGHS only, so CI does not require a commercial solver.

## Run

```bash
python -m decompopt.benders_facility
python -m decompopt.column_generation
python -m decompopt.lagrangian_knapsack
python -m decompopt.progressive_hedging
python -m decompopt.column_constraint_generation\npython -m decompopt.consensus_admm
```

## Research standard

Each native benchmark should report, where applicable:

- primal incumbent;
- valid lower/upper bound;
- optimality or duality gap;
- iteration history;
- independent feasibility checks;
- comparison with an extensive-form or exact reference solve on a small instance.

The branch-and-price benchmark uses a set-partitioning master with exact Ryan-Foster pair branching. Its small pricing oracle enumerates feasible bin patterns so branch restrictions are applied to generated and not-yet-generated columns alike; this is deliberately verification-oriented rather than a claim of large-instance scalability. The ADMM benchmark additionally reports primal and dual residuals, stopping tolerances, objective gap, and distance to an independently computed centralized optimum. Each agent factorizes its local quadratic system once and reuses that factorization across iterations.\n\nThe repository does not treat iteration count alone as evidence of scalability. Serious large-scale studies should additionally report formulation size, hardware, solver/runtime version, wall time, memory-relevant dimensions, stopping rules, and final bounds.

## Roadmap

Planned extensions include:

- logic-based Benders;
- Dantzig-Wolfe decomposition beyond cutting stock;
- branch-and-price;
- decomposition for stochastic and distributionally robust models;
- stabilization and cut/column management strategies.

## License

Research and educational use. Add a repository-level license before redistributing or using the code under a specific licensing regime.
