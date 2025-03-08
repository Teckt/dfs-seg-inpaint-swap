import os

TEXT_ENCODER_CLIP_L_PATH='openai/clip-vit-large-patch14'
FLUX_FILL_HYPER_PATH='flux-fill-fp8'
FLUX_HYPER_PATH='flux-fp8'
FLUX_FILL_PATH='black-forest-labs/FLUX.1-Fill-dev'
FLUX_PATH='black-forest-labs/FLUX.1-dev'
COG_PATH='THUDM/CogVideoX-5b-I2V'
USE_OPTIMUM_QUANTO=True
USE_CPU_OFFLOAD=False
USE_SEQUENTIAL_CPU_OFFLOAD=False
FUSE_HYPER=False
FUSE_HYPER_ALPHA=0.125
FUSE_HYPER_REPO='ByteDance/Hyper-SD'
FUSE_HYPER_MODEL_FILE='Hyper-FLUX.1-dev-8steps-lora.safetensors'
DOWNLOAD_FILES=True
# the dir to look for lora files
LORA_PATH="C:\\Users\\teckt\\PycharmProjects\\kohya\\kohya_ss\\training_data\\model"
# the default name to give for every output image file
REPAINT_OUTPUT_DIR="redressed_results"
IMAGE_OUTPUT_DIR="generated_results"
OUTPUT_FILE_BASE_NAME="outputImage.png"
CRED_PATH='firebase_cred.json'
STORAGE_BUCKET_ID='df-studio-1.appspot.com'
JOB_DIR = os.path.join('C:' + os.sep, 'deepfakes', 'df-maker-files')