#!/usr/bin/env bash

/usr/bin/env

jupytext --output CG_water_difftre.ipynb CG_water_difftre.md
jupyter nbconvert --to ipynb --inplace --execute --allow-errors CG_water_difftre.ipynb
cp CG_water_difftre.ipynb ../docs/examples/CG_water_difftre.ipynb
