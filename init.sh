#!/bin/bash
echo "Installing dependencies"
conda install --file requirements.txt -c pytorch -y > installation.log
echo "Compiling CPU and GPU kernels"
cd torch_c/sc-extension
./compile_all.sh > ../../compilation.log
cd ../../