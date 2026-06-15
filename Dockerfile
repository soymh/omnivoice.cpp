# omnivoice-server — multi-stage CUDA Docker build.
#
# Build:
#   docker build -t omnivoice-server .
#
# Run:
#   docker run --gpus all -v ./models:/models -p 127.0.0.1:8080:8080 omnivoice-server
#
# Override command to change model paths or flags:
#   docker run --gpus all -v ./models:/models omnivoice-server \
#     --model /models/my-model.gguf \
#     --codec /models/my-codec.gguf \
#     --lang en --instruct "male, young adult"

# Stage 1 — build omnivoice-server from source
FROM nvidia/cuda:12.2-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHITECTURES="75-virtual;80-virtual;86-real;89-real"
ARG BUILD_TYPE=Release

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy source tree (excluding large / local data)
COPY CMakeLists.txt .
COPY tools/version.cmake tools/
COPY ggml/ ggml/
COPY vendor/ vendor/
COPY src/ src/
COPY tools/ tools/

RUN mkdir build && cd build && \
    cmake .. \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        && cmake --build . --config "${BUILD_TYPE}" -j "$(nproc)" --target omnivoice-server

# Stage 2 — minimal runtime image
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/omnivoice-server /usr/local/bin/omnivoice-server

EXPOSE 8080

ENTRYPOINT ["omnivoice-server"]
CMD ["--model", "/models/omnivoice-base-Q8_0.gguf", \
     "--codec", "/models/omnivoice-tokenizer-Q8_0.gguf", \
     "--host", "0.0.0.0", "--port", "8080"]
