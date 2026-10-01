FROM python:3.13-slim

# Libraries headless Blender still loads, even with --background
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates xz-utils \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 \
        libsm6 libice6 libgl1 libegl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Same Blender version as local development. The server runs it as a
# subprocess via BLENDER_PATH; bpy/bmesh/mathutils/numpy come bundled with it.
# download.blender.org rejects some non-browser clients (403), so try a mirror
# first; the pinned official checksum verifies whichever source answers.
ARG BLENDER_VERSION=4.4.3
ARG BLENDER_SHA256=8d3be07d2bc412b502c6bfe3cfe3e22195a4164076867da987ce148d73c27946
RUN f=blender-${BLENDER_VERSION}-linux-x64.tar.xz; \
    for base in https://mirrors.ocf.berkeley.edu/blender/release \
                https://ftp.halifax.rwth-aachen.de/blender/release \
                https://download.blender.org/release; do \
        curl -fsSL -o /tmp/$f $base/Blender4.4/$f && break; \
    done \
    && echo "${BLENDER_SHA256}  /tmp/$f" | sha256sum -c - \
    && tar -xJf /tmp/$f -C /opt && rm /tmp/$f \
    && ln -s /opt/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender \
    && blender --background --factory-startup --version
ENV BLENDER_PATH=/usr/local/bin/blender

# SAM2 weights aren't in git; fetch the same checkpoint seg_server.py loads
RUN mkdir -p /app/checkpoints \
    && curl -fsSL -o /app/checkpoints/sam2.1_hiera_small.pt \
         https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# CPU-only torch (Railway has no GPU). The env var also reaches the isolated
# build env pip creates for sam2, which needs torch to build; SAM2_BUILD_CUDA=0
# skips its optional CUDA extension.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    SAM2_BUILD_CUDA=0 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "seg_server.py"]
