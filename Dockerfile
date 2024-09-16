FROM docker.io/zikast/mrtrixfs:1.3

# Updated Dockerfile with some extra packages and Intel Python

# Remove libtorch_cuda, which is not needed and takes up a lot of space:
RUN \
    find /opt -name libtorch_cuda.so -exec truncate -s 0 {} \;

# Install some extra packages:
RUN \
  export DEBIAN_FRONTEND="noninteractive" && \
  apt-get update -q && \
  apt-get install -yq --no-install-recommends ca-certificates curl gawk moreutils time && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Intel Python:
RUN \
    conda run -n base pip install --no-cache-dir --upgrade sh fsspec s3fs loguru tqdm && \
    conda clean -y --all --force-pkgs-dirs

# Change entrypoint for easier debugging:
ENTRYPOINT ["/bin/bash","-l"]
