#!/usr/bin/env bash

# Disable everything related to fusions to potentially avoid issue
# common_utilization <= producer_output_utilization

# export XLA_FLAGS=""

/usr/bin/env

jupytext --output ../docs/examples/CG_alanine_dipeptide.ipynb CG_alanine_dipeptide.md
cd ../docs/examples && jupyter nbconvert --to ipynb --inplace --execute CG_alanine_dipeptide.ipynb
cp CG_alanine_dipeptide.ipynb ../../examples/CG_alanine_dipeptide.ipynb
