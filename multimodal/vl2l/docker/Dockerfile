# Replace with proper vllm container
FROM nvcr.io/nvidian/vllm-cuda13



# 0. Install uv
# The fastest way to get uv is to copy it from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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
RUN uv pip install --system --no-cache git+https://github.com/CentML/mlperf-inference.git@b35b0579774a6cfc56c4383802ef4f6fb24ae698#subdirectory=multimodal/vl2l

# 4. Re-install Flash Attention
# uv handles uninstalls/upgrades automatically, so explicit uninstall is rarely needed.
# --no-build-isolation is crucial for flash-attn to find the torch libraries.
RUN uv pip install --system --no-cache --no-build-isolation flash-attn==2.8.3

# Set the entrypoint to bash so it opens a shell by default
ENTRYPOINT ["/bin/bash"]