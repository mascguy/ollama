# vim: filetype=dockerfile

ARG FLAVOR=${TARGETARCH}

ARG GO_VERSION=1.23.4
ARG CUDA_11_VERSION=11.3
ARG CUDA_12_VERSION=12.4
ARG ROCM_VERSION=6.1.2
ARG JETPACK_5_VERSION=r35.4.1
ARG JETPACK_6_VERSION=r36.2.0
ARG CMAKE_VERSION=3.31.2

FROM --platform=linux/amd64 rocm/dev-centos-7:${ROCM_VERSION}-complete AS base-amd64
RUN sed -i -e 's/mirror.centos.org/vault.centos.org/g' -e 's/^#.*baseurl=http/baseurl=http/g' -e 's/^mirrorlist=http/#mirrorlist=http/g' /etc/yum.repos.d/*.repo \
    && yum install -y yum-utils devtoolset-10-gcc devtoolset-10-gcc-c++ \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo \
    && curl -s -L https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz | tar -Jx -C /usr/local/bin --strip-components 1
ENV PATH=/opt/rh/devtoolset-10/root/usr/bin:/opt/rh/devtoolset-11/root/usr/bin:$PATH

FROM --platform=linux/arm64 rockylinux:8 AS base-arm64
# install epel-release for ccache
RUN yum install -y yum-utils epel-release \
    && yum install -y clang ccache \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
ENV CC=clang CXX=clang++

FROM base-${TARGETARCH} AS base
ARG CMAKE_VERSION
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml

FROM base AS cpu
RUN if [ "$(uname -m)" = "x86_64" ]; then yum install -y devtoolset-11-gcc devtoolset-11-gcc-c++; fi
ENV PATH=/opt/rh/devtoolset-11/root/usr/bin:$PATH
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'Default' && cmake --build --parallel --preset 'Default'

FROM base AS cuda-11
ARG CUDA_11_VERSION
RUN yum install -y cuda-toolkit-${CUDA_11_VERSION//./-}
ENV PATH=/usr/local/cuda-11/bin:$PATH
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CUDA 11' && cmake --build --parallel --preset 'CUDA 11'

FROM base AS cuda-12
ARG CUDA_12_VERSION
RUN yum install -y cuda-toolkit-${CUDA_12_VERSION//./-}
ENV PATH=/usr/local/cuda-12/bin:$PATH
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CUDA 12' && cmake --build --parallel --preset 'CUDA 12'

FROM base AS rocm-6
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'ROCm 6' && cmake --build --parallel --preset 'ROCm 6'

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK_5_VERSION} AS jetpack-5
ARG CMAKE_VERSION
RUN apt-get update && apt-get install -y curl ccache \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'JetPack 5' && cmake --build --parallel --preset 'JetPack 5'

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK_6_VERSION} AS jetpack-6
ARG CMAKE_VERSION
RUN apt-get update && apt-get install -y curl ccache \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'JetPack 6' && cmake --build --parallel --preset 'JetPack 6'

FROM base AS build
ARG GO_VERSION
RUN curl -fsSL https://golang.org/dl/go${GO_VERSION}.linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH
WORKDIR /go/src/github.com/ollama/ollama
COPY . .
ENV GOFLAGS="-trimpath -buildmode=pie -ldflags=-s"
ENV CGO_ENABLED=1
RUN --mount=type=cache,target=/root/.cache/go-build \
    go build -o /bin/ollama .

FROM --platform=linux/amd64 scratch AS amd64
COPY --from=cuda-11 dist/build/lib/libggml-cuda.so /lib/libggml-cuda-11.so
COPY --from=cuda-12 dist/build/lib/libggml-cuda.so /lib/libggml-cuda-12.so

FROM --platform=linux/arm64 scratch AS arm64
COPY --from=cuda-11 dist/build/lib/libggml-cuda.so /lib/libggml-cuda-11.so
COPY --from=cuda-12 dist/build/lib/libggml-cuda.so /lib/libggml-cuda-12.so
COPY --from=jetpack-5 dist/build/lib/libggml-cuda.so /lib/libggml-jetpack-5.so
COPY --from=jetpack-6 dist/build/lib/libggml-cuda.so /lib/libggml-jetpack-6.so

FROM --platform=linux/arm64 scratch AS rocm
COPY --from=rocm-6 dist/build/lib/libggml-hip.so /lib/libggml-hip.so

FROM ${FLAVOR} AS archive
COPY --from=cpu dist/build/lib/libggml-base.so /lib/
COPY --from=cpu dist/build/lib/libggml-cpu-*.so /lib/
COPY --from=build /bin/ollama /bin/ollama

FROM ubuntu:20.04
RUN apt-get update \
    && apt-get install -y ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=archive /lib/ /lib/
COPY --from=archive /bin/ /bin/
EXPOSE 11434
ENV OLLAMA_HOST=0.0.0.0
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
