import os

TEXT_ENCODER_CLIP_L_PATH='openai/clip-vit-large-patch14'
FLUX_FILL_PATH='black-forest-labs/FLUX.1-Fill-dev'
FLUX_PATH='black-forest-labs/FLUX.1-dev'
COG_PATH='THUDM/CogVideoX-5b-I2V'  # cogxvideo repo
USE_OPTIMUM_QUANTO=False  # quantize with optimum quanto
USE_CPU_OFFLOAD=True  # vram savings
USE_SEQUENTIAL_CPU_OFFLOAD=True  # bigger vram savings
FUSE_HYPER=False  # whether to fuse hyper lora with transformer
FLUX_FILL_HYPER_PATH='flux-fill-fp8'  # the pretrained path to the fused model
FLUX_HYPER_PATH='flux-fp8'  # the pretrained path to the fused model
FUSE_HYPER_ALPHA=0.125  # set alpha for turbo lora
FUSE_HYPER_REPO='ByteDance/Hyper-SD'  #repo to hyper model
FUSE_HYPER_MODEL_FILE='Hyper-FLUX.1-dev-8steps-lora.safetensors'  #file name to hyper model
USE_LOCAL_FILES=False  # whether to download models
LORA_PATH="C:\\Users\\teckt\\PycharmProjects\\kohya\\kohya_ss\\training_data\\model" # the dir to look for lora files
REPAINT_OUTPUT_DIR="redressed_results"  # dir to store repainted images
IMAGE_OUTPUT_DIR="generated_results"  # dit to store generated images
OUTPUT_FILE_BASE_NAME="outputImage.png" # the default name to give for every output image file
# CRED_PATH='C:/Users/teckt/PycharmProjects/iae_dfstudio/df-studio-1-25ff59cab415.json'
CRED_PATH='firebase_cred.json'  # firebase credentials
STORAGE_BUCKET_ID='df-studio-1.appspot.com'  # fire storage bucket id
JOB_DIR = "df-maker-files"  # the directory to store DFS files