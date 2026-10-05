# llama.cpp 11fe02151 (the ghcr.io/ggml-org/llama.cpp:full build of this campaign) rebuilt with the
# official CPU flags plus one define: Connection: close after every response (cpp-httplib keep-alive max 1);
# reused pooled benchmark connections raced a silent server-side close (ServerDisconnected).
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ggml-org/llama.cpp /src && cd /src && git checkout 11fe02151
WORKDIR /src
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DCPPHTTPLIB_KEEPALIVE_MAX_COUNT=1" \
    && cmake --build build -j 32 --target llama-server
RUN mkdir -p /app && cp build/bin/llama-server /app/ && find build -name "*.so*" -exec cp -P {} /app/ \;

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /app /app
ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["/app/llama-server"]
