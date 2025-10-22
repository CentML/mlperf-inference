# Start from the slim image you want
FROM python:3.12-slim

# Switch to root to install system packages
USER root

# Install build-essential, which includes g++ (C++ compiler) and make
RUN apt-get update && \
    apt-get install -y build-essential && \
    # Clean up the apt cache to keep the image small
    rm -rf /var/lib/apt/lists/*