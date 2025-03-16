# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

ARG GEMINI_KEY

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    git-lfs \
    wget \
    libgl1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install comfy-cli
RUN pip install comfy-cli

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 11.8 --nvidia --version 0.3.23

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install runpod
RUN pip install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy the snapshot file
ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
RUN /restore_snapshot.sh

# RUN  echo "google_api_key=${GEMINI_KEY}" >> comfyui/custom_nodes/ComfyUI_LayerStyle_Advance/api_key.ini

WORKDIR /comfyui/custom_nodes/comfyui_layerstyle

RUN pip3 install -r requirements.txt

WORKDIR /comfyui/custom_nodes/ComfyUI_LayerStyle_Advance

RUN pip3 install -r requirements.txt


WORKDIR /comfyui/custom_nodes/comfyui-impact-pack

RUN pip3 install -r requirements.txt

WORKDIR /comfyui/custom_nodes/comfyui-impact-subpack

RUN pip3 install -r requirements.txt


WORKDIR  /comfyui/custom_nodes/comfyui_pulid_flux_ll
RUN pip3 install -r requirements.txt



RUN pip3 install piexif

RUN pip3 install facexlib

RUN pip3 install insightface==0.7.3

RUN pip3 install torchscale


# # Start container
# CMD ["/start.sh"]

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN


# Change working directory to ComfyUI
WORKDIR /comfyui

RUN git lfs install
# Create necessary directories
RUN mkdir -p models/checkpoints models/clip models/loras models/pulid models/insightface/models/antelopev2 models/facexlib models/EVF-SAM/evf-sam models/upscale_models models/clip_seg/clipseg-rd64-refined models/unet models/vae models/ultralytics/bbox

RUN git clone https://huggingface.co/MonsterMMORPG/tools models/insightface/models/antelopev2

RUN git clone https://huggingface.co/YxZhang/evf-sam models/EVF-SAM/evf-sam

RUN git clone https://huggingface.co/CIDAS/clipseg-rd64-refined models/clip_seg/clipseg-rd64-refined

# Download checkpoints/vae/LoRA to include in image based on model type
# RUN wget -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors && \
#     wget -O models/pulid/pulid_flux_v0.9.1.safetensors https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors && \
#     wget -O models/clip/EVA02_CLIP_L_336_psz14_s6B.pt https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt && \
#     wget -O models/upscale_models/RealESRGAN_x2.pth https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth && \
#     wget -O models/loras/PIXAR_4.0.safetensors https://huggingface.co/alexanderburuma/pixar4/resolve/main/PIXAR_4.0.safetensors && \
#     wget -O models/facexlib/parsing_bisenet.pth https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth && \
#     wget -O models/facexlib/parsing_parsenet.pth https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth && \
#     wget -O models/facexlib/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth;

# USing GGUF
RUN wget -O models/unet/flux1-dev-Q4_0.gguf https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_0.gguf && \
    wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
    wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
    wget -O models/vae/flux_ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors && \
    wget -O models/pulid/pulid_flux_v0.9.1.safetensors https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors && \
    wget -O models/clip/EVA02_CLIP_L_336_psz14_s6B.pt https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt && \
    wget -O models/upscale_models/RealESRGAN_x2.pth https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth && \
    wget -O models/loras/PIXAR_4.0.safetensors https://huggingface.co/alexanderburuma/pixar4/resolve/main/PIXAR_4.0.safetensors && \
    wget -O models/ultralytics/bbox/face_yolov8m.pt https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt && \
    wget -O models/ultralytics/bbox/hand_yolov8s.pt https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt && \
    wget -O models/facexlib/parsing_bisenet.pth https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth && \
    wget -O models/facexlib/parsing_parsenet.pth https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth && \
    wget -O models/facexlib/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth;
# Stage 3: Final image
FROM base as final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start container
CMD ["/start.sh"]