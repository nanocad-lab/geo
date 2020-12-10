#!/bin/bash
echo "Creating environment"
conda create --name geo --file requirements.txt -c pytorch -y > installation.log
eval "$(conda shell.bash hook)"
conda activate geo
echo "Compiling CPU and GPU kernels"
cd torch_c/sc-extension
./compile_all.sh > ../../compilation.log
cd ../../