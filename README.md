Computes strict bounds intervals for regional fluxes. The optimization is implemented
using an ADMM algorithm, using `scipy` as the sub-optimization backend and wrapping around
GEOS-Chem and GEOS-Chem Adjoint for forward and adjoint evaluations.

# Operational Instructions
## 1 - Create affine correction
Because the ADMM algorithm assumes a linear forward model and GEOS-Chem for CO2 transport is affine in the scaling factors, we must compute a correction accounting for the contribution to XCO2 as a result of non-biospheric sources of CO2.

To generate this correction vector, run
```bash
python create_affine_correction.py
```
The above saves the output file to `/glade/work/mcstanley/admm_objects/fixed_optimization_inputs/affine_correction.npy`.

## Run endpoint optimizations
LEP is run using the bash script `run_lep.sh` and predictably, the UEP optimization is run using `run_uep.sh`. Both of these run the script `run_endpoint_opt_carbon_flux.py` by changing key environmental variables.

# Required directory structure
## Carbon Flux
>>

## Unfolding Experiment
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

# Certification of Components
1. Affine Correction Generation
    1. __Performed__ on 6/13/2023. File saved to `glade/work/mcstanley/admm_objects/fixed_optimization_inputs/affine_correction.npy` by running `python create_affine_correction.py` in Cheyenne.
    2. Involves correct functionality of `forward_adjoint_evaluators.forward_eval_cf`.
2. Linear Component of Forward model
    1. `forward_adjoint_evaluators.forward_linear_eval_cf`
3. File I/O for adjoint model wrapper
    1. Requires reading from existing GOSAT observation directory and creating a new faux-GOSAT observation in which to embed a candidate w vector.
    2. Tested in `w_gen_utils.py`.