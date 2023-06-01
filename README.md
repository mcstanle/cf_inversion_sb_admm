Computes strict bounds intervals for regional fluxes. The optimization is implemented
using an ADMM algorithm, using `scipy` as the sub-optimization backend and wrapping around
GEOS-Chem and GEOS-Chem Adjoint for forward and adjoint evaluations.

# Operational Instructions
Stay tuned!

# Required directory structure
Create the following directory structure:
```bash
├── data
│   ├── lep_diagnostic_plots
│   │   ├── LEP_mu1e3_20maxiter_12suboptiter.png
│   │   ├── LEP_mu1e3_20maxiter_12suboptiter_feas.png
│   │   └── LEP_mu1e3_20maxiter_12suboptiter_opt.png
│   ├── uep_diagnostic_plots
│   │   ├── UEP_mu1e1_20maxiter_12suboptiter.png
│   │   ├── UEP_mu1e1_20maxiter_12suboptiter_feas.png
│   │   ├── UEP_mu1e1_20maxiter_12suboptiter_opt.png
│   │   ├── UEP_mu1e2_20maxiter_12suboptiter.png
│   │   ├── UEP_mu1e2_20maxiter_12suboptiter_feas.png
│   │   ├── UEP_mu1e2_20maxiter_12suboptiter_opt.png
│   │   ├── UEP_mu1e3_20maxiter_12suboptiter.png
│   │   ├── UEP_mu1e3_20maxiter_12suboptiter_feas.png
│   │   ├── UEP_mu1e3_20maxiter_12suboptiter_opt.png
│   │   ├── UEP_mu1e5_20maxiter_12suboptiter.png
│   │   ├── UEP_mu1e5_20maxiter_12suboptiter_feas.png
│   │   └── UEP_mu1e5_20maxiter_12suboptiter_opt.png
│   ├── unfolding_data.pkl
│   ├── unfolding_results_LEP.pkl
│   └── unfolding_results_UEP.pkl
```
`data` holds the actual optimzer results. `lep_diagnostic_plots` and `uep_diagnostic_plots` are save locations for the outputted feasibility and optimality (in available) plots. `adjoint_hash_tables` stores the hash tables allowing for the adjoint mapping lookups.