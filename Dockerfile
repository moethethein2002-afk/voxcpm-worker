# VoxCPM2 requires torch>=2.5.0 (see https://voxcpm.readthedocs.io/en/latest/installation.html)
# The previous base image shipped torch 2.4.0, which is below the minimum
# required version and caused the worker to crash on startup (exit code 1).
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 ships torch 2.8.0 + CUDA 12.8.1.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Install ffmpeg & git (git is needed by some voxcpm/transformers download paths)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install dependencies. Do NOT use --no-deps: voxcpm needs transformers,
# torchcodec, huggingface-hub, safetensors, etc. Skipping them causes
# ModuleNotFoundError at import time (another cause of the crash loop).
COPY requirements.txt /workspace/requirements.txt
# NOTE: --ignore-installed used to be applied to the WHOLE requirements.txt
# install here. That's what was causing the RecursionError crash below:
# --ignore-installed tells pip to treat every package (including numpy,
# which the base image already ships) as "not installed" and lay down a
# fresh copy on top, WITHOUT properly removing the base image's existing
# numpy first. That left two partially-overlapping numpy installs on disk
# (mismatched .py/.so files from two different numpy builds), and importing
# that Frankenstein numpy is exactly what produces infinite recursion in
# numpy/core/_dtype.py when it tries to repr a dtype (RecursionError: maximum
# recursion depth exceeded).
# Fix: scope --ignore-installed to ONLY the untracked system "cryptography"
# package it was actually meant for, and let pip manage numpy/torch/etc.
# normally so it can cleanly upgrade/replace them.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed cryptography && \
    pip install --no-cache-dir -r requirements.txt

# Force a clean reinstall of torch/torchaudio.
# Even though the base image already ships torch/torchaudio 2.8.0, pip sees
# "requirement already satisfied" for these two packages and skips them —
# so if the base image's torchaudio install is broken/duplicated (multiple
# _torchaudio*.so files causing "libtorchaudio.so: cannot open shared object
# file"), that broken install never gets repaired. --force-reinstall wipes
# and reinstalls a clean, single-source copy. --no-deps avoids re-triggering
# the full dependency resolver (which caused long backtracking on gradio
# versions during the main requirements.txt install above).
RUN pip install --no-cache-dir --force-reinstall --no-deps torch==2.8.0 torchaudio==2.8.0

# Explicitly uninstall numpy/scipy first, THEN reinstall clean copies.
# force-reinstall alone can still leave orphan files behind if a previous
# install (e.g. one laid down by --ignore-installed, or the base image's own
# copy) wasn't fully tracked by pip's RECORD — uninstalling first guarantees
# there is exactly one numpy/scipy on disk afterwards. --no-deps on both
# stops scipy's dependency resolution from silently pulling in yet another
# numpy build.
RUN pip uninstall -y numpy scipy && \
    pip install --no-cache-dir --no-deps numpy==1.26.4 scipy==1.13.1

# Build-time sanity check: fail the build loudly instead of at runtime if
# numpy is broken/duplicated or can't do a basic dtype repr.
RUN python -c "import numpy; print('numpy', numpy.__version__, numpy.__file__); print(numpy.dtype('float32'))"

COPY handler.py /workspace/handler.py

# Optional: bake the model into the image at build time so cold starts don't
# need to hit Hugging Face. Comment out if you'd rather use a Network Volume
# (mounted at /runpod-volume) or download on first request.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2', local_dir='/models/VoxCPM2')"
ENV VOXCPM_MODEL_DIR=/models/VoxCPM2

CMD ["python", "-u", "/workspace/handler.py"]
