## Installation

Most of the requirements of this projects are exactly the same as [Scene Graph Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [Neural-Backed Decision Trees](https://github.com/alvinwan/neural-backed-decision-trees). If you have any problem of your environment, you should check the [issues page of SG Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/issues) and [issues page of NBDT](https://github.com/alvinwan/neural-backed-decision-trees/issues) first.

### Requirements:
- PyTorch >= 1.2
- torchvision >= 0.4
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash

conda create --name scene_graph_benchmark
conda activate scene_graph_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install scipy
conda install h5py

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.0
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/CYVincent/Scene-Graph-Transformer-CogTree.git
cd Scene-Graph-Transformer-CogTree/sg_benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

cd $INSTALL_DIR
cd Scene-Graph-Transformer-CogTree/nbdt
python setup.py develop

unset INSTALL_DIR 

