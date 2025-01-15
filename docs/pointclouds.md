## MinkowskiEngine Installation

### On Snellius

`conda` should be installed first. This part doesn't require GPU machine and only done once.

```bash
module load 2023
module load Anaconda3/2023.07-2
conda init
```

After `conda init`, the terminal should be closed. So be careful, if you allocated the GPU, you will loss your GPUs, unless you do `ssh` to the allocated machine. E.g. `ssh gcn14`. 

> `MinkowskiEngine` requires GPU. You can ask for GPU by `salloc`.

#### CUDA 11

```bash
salloc --gpus=1 --partition=gpu --time=02:00:00

# MinkowskiEngine==0.5.4
conda create -n mink2 python=3.8
python activate mink2
module load 2022
module load CUDA/11.7.0 # CUDA/11.6.0 was giving error
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install openblas-devel -c anaconda

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

#### CUDA 12

Only runs on H100. `MinkowskiEngine` doesn't work before some changes in the `MinkowskiEngine` code. This fix is described here.

**Fix**: There is an issue described in [here](https://github.com/NVIDIA/MinkowskiEngine/issues/594#issuecomment-2294860926). This fix is based on [this link](https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-1773458776).

- Get the code
```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git 
cd MinkowskiCu12
```

- Fix the code
```bash
1 - .../MinkowskiEngine/src/convolution_kernel.cuh

Add header:
#include <thrust/execution_policy.h>

2 - .../MinkowskiEngine/src/coordinate_map_gpu.cu

Add headers:

#include <thrust/unique.h>
#include <thrust/remove.h>

3 - .../MinkowskiEngine/src/spmm.cu

Add headers:

#include <thrust/execution_policy.h>
#include <thrust/reduce.h> 
#include <thrust/sort.h>

4 - .../MinkowskiEngine/src/3rdparty/concurrent_unordered_map.cuh

Add header:
#include <thrust/execution_policy.h>
```

- Installation

```bash
salloc --gpus=1 --partition=gpu_h100 --time=02:00:00

module load 2023
module load CUDA/12.1.1
conda create -n mink-cu12 python=3.8
conda activate mink-cu12
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install openblas-devel -c anaconda

cd ~/dev/MinkowskiCu12
# Before this command apply the Fix. 
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

#### Docker CUDA 12 with the Fix 

This information is from [here](https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-2371637255).

The base image is `nvidia/cuda:12.1.1-devel-ubuntu20.04` with `PyTorch 2.3.1` and `CUDA 12.1`.

```bash
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine \
    && cd /tmp/MinkowskiEngine \
    && sed -i '31i #include <thrust/execution_policy.h>' ./src/convolution_kernel.cuh \
    && sed -i '39i #include <thrust/unique.h>\n#include <thrust/remove.h>' ./src/coordinate_map_gpu.cu \
    && sed -i '38i #include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' ./src/spmm.cu \
    && sed -i '38i #include <thrust/execution_policy.h>' ./src/3rdparty/concurrent_unordered_map.cuh \
    && python setup.py install --force_cuda --blas=openblas \
    && cd - \
    && rm -rf /tmp/MinkowskiEngine
```

#### Test MinkowskiEngine

```python
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import sys
sys.path.insert(0,"~/dev/MinkowskiEngine/tests/python")
from common import data_loader

class ExampleNetwork(ME.MinkowskiNetwork):
    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(128, out_feat)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        return self.linear(out)

criterion = nn.CrossEntropyLoss()
net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
print(net)

# a data loader must return a tuple of coords, features, and labels.
coords, feat, label = data_loader()
input = ME.SparseTensor(feat, coordinates=coords)
# Forward
output = net(input)

# Loss
loss = criterion(output.F, label)
```

## Install other packages For SegmentAnyTree

```bash
# open3d, torch-sparse: very slow
# torch-points-kernels: different version of numpy, sklearn
pip install -U pip
pip install hydra-core wandb tqdm gdown types-six types-requests h5py
pip install numpy scikit-image numba 
pip install matplotlib
pip install plyfile
pip install torchnet tensorboard
pip install open3d
pip install torch-scatter torch-sparse torch-cluster torch-geometric pytorch_metric_learning torch-points-kernels
```

## Miscellaneous 

#### GPU Compute Capability

You can define from the below code or from [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).

```python
import torch
torch.cuda.get_device_capability()
```

H100: 9.0, A100: 8.6

#### CUDA modules available on Snellius

```bash
# module load 2024
CUDA/12.6.0
cuDNN/9.5.0.50-CUDA-12.6.0

# module load 2023
CUDA/12.1.1 
CUDA/12.4.0
cuDNN/8.9.2.26-CUDA-12.1.1

# module load 2022
CUDA/11.6.0    
CUDA/11.7.0
CUDA/11.8.0
cuDNN/8.4.1.50-CUDA-11.7.0
cuDNN/8.6.0.163-CUDA-11.8.0
```

#### CUDA modules paths on Snellius

To get the path information such as `CUDA_HOME`, `CUDA_PATH`, `PAHT`, `LIBRARY_PATH`, `LD_LIBRARY_PATH` use: 

```bash
module load CUDA/12.6.0
module display CUDA/12.6.0
```


### Tried but failed

These are my previos attemps to solve issues encountered. But resolvinig issues didn't install MinkowskiEngine. But it is worth mentioning them here.

#### pip DEPRECATION: --build-option and --global-option

This option doesn't work, even with the `--no-build-isolation` fix.

```bash
export CXX=g++
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-build-isolation -v --config-settings=build_option=--force_cuda --config-settings=blas_include_dirs=${CONDA_PREFIX}/include --config-settings=blas=openblas
```

MinkowskiEngine requires torch at build time, pip’s default build isolation can cause “Pytorch not found” errors. `--no-build-isolation` tells pip not to create a temporary/isolated environment at build time and instead to use the packages in your current conda/pip environment (where you have PyTorch installed).

#### Distutil issue with Numpy

`numpy.distutils` removed after numpy=1.24.

```bash
conda install "numpy=1.23.*" setuptools pip wheel
```

#### Find openbals

```bash
OPENBLAS=$(find ~/.conda/envs/test -name "libopenblas.so" | head -n 1)
export LD_LIBRARY_PATH=$(dirname $OPENBLAS):$LD_LIBRARY_PATH
pip install --no-cache-dir --force-reinstall numpy
```

```bash
# >>> np.__config__.show()
  "Build Dependencies": {
    "blas": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.28",
      "detection method": "pkgconfig",
      "include directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/include",
      "lib directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.28  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
      "pc file directory": "/project/.openblas"
    },
    "lapack": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.28",
      "detection method": "pkgconfig",
      "include directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/include",
      "lib directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.28  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
      "pc file directory": "/project/.openblas"
    }
  }
```

#### CUDA installation in Docker

```bash
# docker pull or From
# nvidia/cuda:12.6.0-runtime-ubuntu22.04
# nvidia/cuda:12.6.0-devel-ubuntu22.04
# nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# https://developer.nvidia.com/cuda-downloads
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sh cuda_11.1.1_455.32.00_linux.run --silent --toolkit

export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.1
```

### References

- SegmentAnyTree: [paper](https://github.com/SmartForest-no/SegmentAnyTree)
- TreeLearn: [github](https://github.com/ecker-lab/TreeLearn/tree/main), [paper](https://www.sciencedirect.com/science/article/pii/S1574954124004308)
