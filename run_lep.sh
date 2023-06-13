# Run the LEP optimization
#
# Author        : Mike Stanley
# Created       : Jun 13, 2023
# Last Modified : Jun 13, 2023

# define operational parameters
LEP = true
MAX_ITERS = 20
SUBOPT_ITERS = 12

# make variables accessible to python
export LEP
export MAX_ITERS
export SUBOPT_ITERS

# run optimization
python run_endpoint_opt_carbon_flux.py
