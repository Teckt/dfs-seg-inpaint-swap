USE_LOCAL_FILES = False  # whether to download models
TEXT_ENCODER_CLIP_L_PATH = 'openai/clip-vit-large-patch14'
FLUX_FILL_PATH = 'black-forest-labs/FLUX.1-Fill-dev'
FLUX_PATH = 'black-forest-labs/FLUX.1-dev'
COG_PATH = 'THUDM/CogVideoX-5b-I2V'  # cogxvideo repo

USE_CPU_OFFLOAD = True  # vram savings
USE_SEQUENTIAL_CPU_OFFLOAD = False  # bigger vram savings

USE_CUSTOM_FLUX = True
FLUX_CUSTOM_PATH = "Anyfusion/flux-nf4"  # the pretrained path to the custom model
FLUX_HYPER_CUSTOM_PATH = "Anyfusion/flux-hyper-nf4"  # the pretrained path to the custom model
USE_CUSTOM_FLUX_FILL = True
FLUX_FILL_CUSTOM_PATH = 'Anyfusion/flux-fill-nf4'  # the pretrained path to the custom fill model
FLUX_FILL_HYPER_CUSTOM_PATH = 'Anyfusion/flux-fill-hyper-nf4'  # the pretrained path to the custom fill model

LORA_PATH = "C:\\Users\\teckt\\PycharmProjects\\kohya\\kohya_ss\\training_data\\model"  # the dir to look for lora files

REPAINT_OUTPUT_DIR = "redressed_results"  # dir to store repainted images
IMAGE_OUTPUT_DIR = "generated_results"  # dit to store generated images
OUTPUT_FILE_BASE_NAME = "outputImage.png"  # the default name to give for every output image file
OUTPUT_MASK_FILE_BASE_NAME = "outputMaskImage.png"  # the default name to give for every copy of the input mask image file
OUTPUT_ORIGINAL_FILE_BASE_NAME = "outputOriginalImage.png"  # the default name to give for every copy of the orignal image file
# CRED_PATH='C:/Users/teckt/PycharmProjects/iae_dfstudio/df-studio-1-25ff59cab415.json'
CRED_PATH = 'firebase_cred.json'  # firebase credentials; os JOIN WITH ".." to use parent dir as root
STORAGE_BUCKET_ID = 'df-studio-1.appspot.com'  # fire storage bucket id
JOB_DIR = "df-maker-files"  # the directory to store DFS files
# JOB_DIR = "C:/deepfakes/df-maker-files"


### DEBUG ONLY ###
USE_BNB = False
USE_OPTIMUM_QUANTO = False  # quantize with optimum quanto

FUSE_HYPER_LORA = False  # whether to fuse hyper lora with transformer
FUSE_HYPER_ALPHA = 0.125  # set alpha for turbo lora
FUSE_HYPER_LORA_REPO = 'ByteDance/Hyper-SD'  # repo to hyper model
FUSE_HYPER_LORA_MODEL_FILE = 'Hyper-FLUX.1-dev-8steps-lora.safetensors'  # file name to hyper model
# FUSE_HYPER_ALPHA=1  # set alpha for turbo lora
# FUSE_HYPER_LORA_REPO= 'alimama-creative/FLUX.1-Turbo-Alpha'  #repo to hyper model
# FUSE_HYPER_LORA_MODEL_FILE= 'diffusion_pytorch_model.safetensors'  #file name to hyper model

SAVE_MODEL = False  # whether to save the fused model to the path
SAVE_MODEL_PATH = "flux-fill-hyper-nf4"
SHARD_SIZE = "32GB"

SAVE_SEG_IMAGES = False  # save and debug mask images
### DEBUG ONLY ###
