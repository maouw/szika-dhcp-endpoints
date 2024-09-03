# Updated container with some extra packages and Intel Python
FROM docker.io/zikast/mrtrixfs:1.3

# Remove libtorch_cuda, which is not needed and takes up a lot of space (doesn't matter for Docker, but does for Apptainer when building the image):
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
    conda update -n base conda && \
    conda install conda-forge::conda-libmamba-solver conda-forge::libmamba conda-forge::libmambapy conda-forge::libarchive && \
    conda config --set allow_non_channel_urls True && \
    conda config --set solver libmamba && \
    conda config --add channels https://software.repos.intel.com/python/conda/ && \
    conda install -n base -c https://software.repos.intel.com/python/conda/ -c conda-forge intelpython3_core 'python>=3.10' && \
    conda run -n base pip install --no-cache-dir --upgrade sh fsspec s3fs loguru tqdm && \
    conda clean --all --force-pkgs-dirs --yes

# Change entrypoint for easier debugging:
ENTRYPOINT ["/bin/bash","-l"]

