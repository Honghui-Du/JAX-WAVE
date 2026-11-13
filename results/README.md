# Results

| File or directory | Contents |
| --- | --- |
| `confirmation_summary.json`, `confirmation_by_seed.csv` | Three-seed reconstruction on known and unseen materials; field errors are fractions |
| `final_evidence_summary.json` | Paired model comparison, material transfer, refinement and timing; aggregate field errors are percentages |
| `screening.csv`, `capacity.json` | Validation comparisons of network width and training coverage |
| `regularization.json` | Numerical and neural errors, control amplitudes and numerical objective-gap bounds |
| `timing.json`, `amortization.json` | Update and query costs, and repeated-query cost estimates |
| `operator_checks.json` | Discrete-operator verification |
| `histories/` | Training and validation curves |
| `budget_benchmark/` | Per-query errors, synchronized timings, case descriptors, aggregate statistics and example fields |
| `cluster_sensitivity/` | Descriptor-deletion sensitivity inputs and statistics |

The online benchmark contains 144 cases paired through 16 descriptors. Its
primary comparison uses 96 heterogeneous cases. Neural measurements retain
three model seeds. The bootstrap resamples descriptors within each material
family. Run `scripts/analyze_budget_benchmark.py` to recompute the statistics.
