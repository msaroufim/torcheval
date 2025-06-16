cd /home/marksaroufim/pytorch

# Clean the build
python setup.py clean

# Set CUDA debug flags explicitly
# export NVCC_FLAGS="-G"
# export CUDA_NVCC_FLAGS="--G"
# export CMAKE_CUDA_FLAGS="-G"

# Rebuild with debug
DEBUG_CUDA=1 DEBUG=1 python setup.py develop
