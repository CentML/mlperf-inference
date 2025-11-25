# Replace with proper vllm container
FROM nvcr.io/nvidian/vllm-cuda13

ARG REPO_URL=https://github.com/CentML/mlperf-inference.git@b35b0579774a6cfc56c4383802ef4f6fb24ae698

# 1. Install Git
# Required for "git+..." dependencies to work.
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# 2. Set Environment Variables
# Setting LD_LIBRARY_PATH here ensures it works during the build process
# and persists when you run the container later.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH

# 3. Install the specific GitHub branch
# We use --system to install into the container's global python environment
RUN uv pip install --system --no-cache "git+${REPO_URL}#subdirectory=multimodal/vl2l"


# Set the entrypoint to bash so it opens a shell by default
ENTRYPOINT ["/bin/bash"]