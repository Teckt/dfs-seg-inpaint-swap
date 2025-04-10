import torch
from controlnet_aux.processor import Processor
from diffusers import ControlNetModel, MotionAdapter, AnimateDiffControlNetPipeline, AutoencoderKL, \
    StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler, \
    DPMSolverMultistepScheduler, DEISMultistepScheduler, UniPCMultistepScheduler, LCMScheduler, StableDiffusionPipeline, AnimateDiffVideoToVideoControlNetPipeline


USE_ANIMATE_LCM = True

lora_path="C:\\Users\\teckt\\Documents\\stable-diffusion-webui-master\\models\\Lora"
embeddings_path="C:\\Users\\teckt\\Documents\\stable-diffusion-webui-master\\embeddings"
lora_settings = {
        "adapter_names": ["ahx_v1", "more_details"],
        "adapter_weights": [0.6, 0.6]
        # "adapter_names": [],
        # "adapter_weights": []
    }

def load_textual_inversions(pipeline):
    return
    
def load_image_loras(pipeline, is_video=True):
    if is_video:
        if USE_ANIMATE_LCM:
            pipeline.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                        adapter_name="lcm-lora")
        else:
            pipeline.load_lora_weights("guoyww/animatediff", weight_name="v3_sd15_adapter.ckpt",
                                       adapter_name="motion-lora")

        if USE_ANIMATE_LCM:
            lora_settings['adapter_names'].append("lcm-lora")
            # lora_settings['adapter_names'].append("pan-left")
            lora_settings['adapter_weights'].append(0.8)
            # lora_settings['adapter_weights'].append(0.8)
        else:
            lora_settings['adapter_names'].append("motion-lora")
            # lora_settings['adapter_names'].append("pan-left")
            lora_settings['adapter_weights'].append(0.8)
            # lora_settings['adapter_weights'].append(0.8)
    # pipeline.load_lora_weights(lora_path, weight_name="koreanDollLikeness.safetensors",
    #                                 adapter_name="koreanDollLikeness"
    #                                 )
    pipeline.load_lora_weights(lora_path, weight_name="ahx_v1.safetensors",
                                    adapter_name="ahx_v1"
                                    )
    pipeline.load_lora_weights(lora_path, weight_name="more_details.safetensors",
                                    adapter_name="more_details"
                                    )
    # pipeline.fuse_loras(1.0)
    # pipeline.load_lora_weights(lora_path, weight_name="add_detail.safetensors",
    #                            adapter_name="add_detail"
    #                            )
    # self.pipeline.load_lora_weights(lora_path, weight_name="MS_Real_LegsUpPresenting.safetensors",
    #                                 adapter_name="MS_Real_LegsUpPresenting"
    #                                 )
    pipeline.set_adapters(**lora_settings)


class SD15PipelineManager:
    TEXT_MODE = 0
    IMAGE_MODE = 1
    USE_ANIMATE_DIFF = 2
    USE_ANIMATE_DIFF_V2V = 3

    def __init__(self, torch_dtype=torch.float16, local_files_only=False, unet_model_memory_limit=2,
                 control_net_model_memory_limit=2):

        self.pipe_settings = {
            "mode": SD15PipelineManager.IMAGE_MODE,
            "use_LCM": True,  # only used for video pipe
            "scheduler": "EulerAncestralDiscreteScheduler",
            "use_inpaint_control_net": True,
            "control_net_id": 'openpose'
        }

        self.local_files_only = local_files_only
        self.torch_dtype = torch_dtype

        self.max_control_nets = control_net_model_memory_limit
        self.max_unets = unet_model_memory_limit

        self.motion_models = {}
        self.inpaint_control_net = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint",
                                                                   torch_dtype=self.torch_dtype,
                                                                   local_files_only=self.local_files_only)
        self.control_net = None
        self.unets = {}
        self.pipe = None
        self.control_p = None  # control net processor
        self.vae = None

    def apply_settings(self, **kwargs):
        '''
        self.pipe_settings = {
            "mode": SD15PipelineManager.USE_IMAGE,
            "use_LCM": True, # only used for video pipe
            "scheduler": "EulerAncestralDiscreteScheduler",
            "use_inpaint_control_net": True,
            "control_net_id": 'openpose'
        }
        '''
        for k, v in kwargs.items():
            if k in self.pipe_settings.keys():
                print(f"Settings ({k}): previous={self.pipe_settings[k]}, new={v}", end="")
                self.pipe_settings[k] = v

    def get_control_net(self, control_net_id):

        print(f"loading control net {control_net_id}")
        # load image processor to annotate images for the control net inputs
        if control_net_id in ['depth_midas', 'canny', 'openpose', 'normal_bae']:
            if self.control_p is not None and self.control_p.processor_id == control_net_id:
                return self.control_net

            self.control_p = Processor(control_net_id)
        else:
            print("select a control from 'depth_midas', 'canny', 'openpose', 'normal_bae'")
            assert False

        # map custom readable shortcuts
        if control_net_id == 'depth_midas':
            pretrained_path = "lllyasviel/control_v11f1p_sd15_depth"
        if control_net_id == 'canny':
            pretrained_path = "lllyasviel/sd-controlnet-canny"
        if control_net_id == 'openpose':
            pretrained_path = 'lllyasviel/control_v11p_sd15_openpose'
        if control_net_id == 'normal_bae':
            pretrained_path = 'lllyasviel/control_v11p_sd15_normalbae'

        control_net = ControlNetModel.from_pretrained(pretrained_path, torch_dtype=self.torch_dtype,
                                                      local_files_only=self.local_files_only)

        return control_net

    def get_motion_model(self):
        print("loading motion adapter model")
        # determine which model to use
        if self.pipe_settings.get("use_LCM", True):
            # needs to be used with LCM scheduler
            motion_id = "wangfuyun/AnimateLCM"
        else:
            motion_id = "https://huggingface.co/Lightricks/LongAnimateDiff/blob/main/lt_long_mm_32_frames.ckpt"

        # Don't load existing model
        if motion_id in self.motion_models:
            return self.motion_models[motion_id]

        if self.pipe_settings.get("use_LCM", True):
            self.motion_models[motion_id] = MotionAdapter.from_pretrained(motion_id, torch_dtype=self.torch_dtype,
                                                                          local_files_only=self.local_files_only)
        else:
            self.motion_models[motion_id] = MotionAdapter.from_single_file(motion_id, torch_dtype=self.torch_dtype,
                                                                           local_files_only=self.local_files_only)

        return self.motion_models[-1]

    def set_pipeline(self):
        mode = self.pipe_settings.get("mode")

        if mode == SD15PipelineManager.IMAGE_MODE:
            self.pipe = self.get_image_pipe()
        elif mode == SD15PipelineManager.TEXT_MODE:
            self.pipe = self.get_image_pipe()
        else:
            self.pipe = self.get_video_pipe()

        # determine if using the inpainting control net
        control_net_id = self.pipe_settings.get("control_net_id", 'openpose')
        if self.pipe_settings.get("use_inpaint_control_net", True):
            # put control nets in list for SD pipeline input
            control_net = self.inpaint_control_net
            # control_net = [
            #     self.inpaint_control_net,
            #     self.get_control_net(control_net_id),
            # ]
        else:
            control_net = self.get_control_net(control_net_id)
        self.pipe.control_net = control_net
        self.pipe.scheduler = self.get_scheduler()
        # # pipeline.to('cuda')
        load_textual_inversions(self.pipe)
        load_image_loras(self.pipe, is_video=False)
        # self.pipe.fuse_lora()

        if mode == SD15PipelineManager.USE_ANIMATE_DIFF or mode == SD15PipelineManager.USE_ANIMATE_DIFF_V2V:
            if self.free_noise_params:
                self.pipe.enable_free_noise(context_length=self.free_noise_params[0],
                                            context_stride=self.free_noise_params[1])
            if self.free_noise_split_params:
                self.pipe.enable_free_noise_split_inference(spatial_split_size=self.free_noise_split_params[0],
                                                            temporal_split_size=self.free_noise_split_params[1])

                # pipeline.unet.enable_attn_chunking(
                #     self.free_noise_params[0])  # Temporal chunking across batch_size x num_frames
                # pipeline.unet.enable_motion_module_chunking(
                #     (512 // 8 // 4) ** 2
                # )  # Spatial chunking across batch_size x latent height x latent width
                # pipeline.unet.enable_resnet_chunking(self.free_noise_params[0])
                # pipeline.unet.enable_forward_chunking(self.free_noise_params[0])

        self.pipe.enable_model_cpu_offload()
        # pipeline.enable_sequential_cpu_offload()
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()

    def get_video_pipe(self, model_id="SG161222/Realistic_Vision_V5.1_noVAE"):
        motion_model = self.get_motion_model()

        if self.pipe_settings.get("mode") == SD15PipelineManager.USE_ANIMATE_DIFF:
            pipeline = AnimateDiffControlNetPipeline.from_pretrained(
                model_id, motion_adapter=motion_model, controlnet=None,
                torch_dtype=self.torch_dtype, local_files_only=self.local_files_only,
                safety_checker=None,
            )
        else:
            pipeline = AnimateDiffVideoToVideoControlNetPipeline.from_pretrained(
                model_id, motion_adapter=motion_model, controlnet=None,
                torch_dtype=self.torch_dtype, local_files_only=self.local_files_only,
                safety_checker=None,
            )
            # Enable FreeNoise for long prompt generation
            # pipeline.enable_free_noise(context_length=self.context_length, context_stride=self.context_stride)
        # auto add vae for some models if vae specified (float16 only)
        if self.torch_dtype == torch.float16 and (
                'novae' in model_id.lower() or 'no-vae' in model_id.lower() or 'no_vae' in model_id.lower()):
            if self.vae is None:
                if self.local_files_only:
                    print(
                        f"Warning: This model is does not include a VAE and you have local_files_only={self.local_files_only}.")
                self.vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16, local_files_only=self.local_files_only
                )
            pipeline.vae = self.vae

        return pipeline

    def get_image_pipe(self, model_id="C:/hf/models/single_file/realisticVisionV60B1_v60B1InpaintingVAE.safetensors"):
        # default can't do anything other than 512x512
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        print("using StableDiffusionControlNetInpaintPipeline")

        if self.pipe_settings.get("mode") == SD15PipelineManager.TEXT_MODE:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id, use_safetensors=True, safety_checker=None,
                torch_dtype=self.torch_dtype, local_files_only=self.local_files_only,
            )
        elif self.pipe_settings.get("use_inpaint_control_net", True):
            pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                model_id, controlnet=self.inpaint_control_net, use_safetensors=True,
                torch_dtype=self.torch_dtype, local_files_only=self.local_files_only,
                safety_checker=None
            )  # requires control image in input
        else:
            pipeline = StableDiffusionInpaintPipeline.from_single_file(
                model_id, controlnet=self.inpaint_control_net, use_safetensors=True,
                torch_dtype=self.torch_dtype, local_files_only=self.local_files_only,
            )

        # auto add vae for some models if vae specified (float16 only)
        if self.torch_dtype == torch.float16 and (
                'novae' in model_id.lower() or 'no-vae' in model_id.lower() or 'no_vae' in model_id.lower()):
            if self.vae is None:
                if self.local_files_only:
                    print(
                        f"Warning: This model is does not include a VAE and you have local_files_only={self.local_files_only}.")
                self.vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16, local_files_only=self.local_files_only
                )
            pipeline.vae = self.vae

        return pipeline

    def get_scheduler(self):
        schedulers = {
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config,
                                                                                           timestep_spacing='linspace',
                                                                                           beta_schedule="linear"),
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                # algorithm_type="dpmsolver++",
                algorithm_type="sde-dpmsolver++",
                solver_order=2,
                use_karras_sigmas=True,
                timestep_spacing='linspace'
            ),
            "DEISMultistepScheduler": DEISMultistepScheduler.from_config(self.pipe.scheduler.config),
            "UniPCMultistepScheduler": UniPCMultistepScheduler.from_config(self.pipe.scheduler.config),
            "LCMScheduler": LCMScheduler.from_config(self.pipe.scheduler.config)
        }

        if self.pipe_settings.get("use_LCM", True):
            return schedulers["LCMScheduler"]
        else:
            scheduler = schedulers.get(self.pipe_settings.get("scheduler"))
            if not scheduler:
                scheduler = schedulers["EulerAncestralDiscreteScheduler"]
            return scheduler

