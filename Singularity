Bootstrap: docker
From: maouw/debug-zikast-mrtrixfs:1.3

%post
    set -eEx
    conda config --add channels https://software.repos.intel.com/python/conda
    conda install -n base -y -c https://software.repos.intel.com/python/conda/ 'python>=3.10' --force-reinstall
    #conda install -n base -y -c conda-forge git git-credential-manager gh
    conda run -n base pip install --no-cache-dir --upgrade sh fsspec s3fs loguru
    conda clean --all --force-pkgs-dirs -y

