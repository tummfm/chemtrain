#!/usr/bin/env bash

# Disable everything related to fusions to potentially avoid issue
# common_utilization <= producer_output_utilization

export XLA_FLAGS="
    --xla_gpu_use_runtime_fusion=false
    --xla_gpu_enable_command_buffer=false
    --xla_gpu_enable_custom_fusions=false
    --xla_gpu_enable_address_computation_fusion=false
    --xla_gpu_enable_triton_softmax_fusion=false
    --xla_gpu_triton_fusion_level=0
    --xla_gpu_enable_priority_fusion=false
    --xla_gpu_enable_triton_softmax_priority_fusion=false
    --xla_gpu_enable_reduction_epilogue_fusion=false
    --xla_gpu_cudnn_gemm_fusion_level=0
    --xla_backend_optimization_level=0
    --xla_gpu_disable_gpuasm_optimizations
    --xla_gpu_force_compilation_parallelism=1
"

/usr/bin/env

jupytext --output ../docs/examples/CG_alanine_dipeptide.ipynb CG_alanine_dipeptide.md
cd ../docs/examples && jupyter nbconvert --to ipynb --inplace --execute --allow-errors CG_alanine_dipeptide.ipynb
cp CG_alanine_dipeptide.ipynb ../../examples/CG_alanine_dipeptide.ipynb
