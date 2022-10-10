# GEO
This is a software library used for training of stochastic computing(SC)-based neural networks.

## Directory Structure
```
.
├── main_conv.py              # Train a CNN in GEO style
├── utils.py                  # Miscellaneous functions
├── network.py                # Model definitions
├── utils_class.py            # Layer definitions
├── utils_functional.py       # SC-specific function definitions
├── utils_own.py              # Training function definitions
├── torch_c                   # C++-based torch extension
    ├── sc-extension          # SC-specific torch extension
        ├── sc.cpp            # Accelerated SC kernels for CPU
        ├── sc_cuda.cpp       # Helper function for accelerated SC kernels for GPU
        ├── sc_cuda_kernel.cu # Accelerated SC kernels for GPU
        ├── setup.py          # Compilation file for CPU kernels
        ├── setup_cuda.py     # Compilation file for GPU kernels
```

## Prerequisites
Ubuntu 18.04 or 20.04 LTS (other distros not tested)

NVIDIA driver >= 440.33

CUDA >= 10.2

conda >= 4.8.4

With conda installed, to create an environment with the required dependencies, execute:
```
chmod +x create_env_init.sh
source create_env_init.sh
```
Or execute the following code to install dependencies:
```
chmod +x init.sh
source init.sh
```

## Usage
To train a neural network in GEO style, execute:
```
python main_conv.py 
```
This trains a 4-layer CNN with partial binary accumulation using linear feedback shift registers (LFSR) as generator. More options can be viewed by executing:
```
python main_conv.py -h
```

Please consider citing our work if you find this useful:
```
@INPROCEEDINGS{tianmu2021geo,
  author={Li, Tianmu and Romaszkan, Wojciech and Pamarti, Sudhakar and Gupta, Puneet},
  booktitle={2021 Design, Automation & Test in Europe Conference & Exhibition (DATE)},
  title={GEO: Generation and Execution Optimized Stochastic Computing Accelerator for Neural Networks},
  year={2021},
  pages={689-694},
  doi={10.23919/DATE51398.2021.9473911}
}
```
