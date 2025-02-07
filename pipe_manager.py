class SD15PipelineManager:
    USE_IMAGE = 0
    USE_ANIMATE_DIFF = 1
    USE_ANIMATE_DIFF_V2V = 2
    def __init__(self, torch_dtype=torch.float16, local_files_only=False, unet_model_memory_limit=2, control_net_model_memory_limit=2):
        
        self.pipe_settings = {
            "mode": SD15PipelineManager.USE_IMAGE,
            "use_LCM": True, # only used for video pipe
            "scheduler": "EulerAncestralDiscreteScheduler",
            "use_inpaint_control_net": True,
            "control_net_id": 'openpose'
        }

        self.local_files_only = local_files_only
        self.torch_dtype = torch_dtype 
        
        self.max_control_nets = control_net_model_memory_limit
        self.max_unets = unet_model_memory_limit

        self.motion_models = {}
        self.control_nets = {}
        self.unets = {}
        self.pipe = None
        self.control_p = None  # control net processor

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
        for k, v in kwargs:
            if k in self.pipe_settings:
                print(f"Settings ({k}): previous={self.pipe_settings[k]}, new={v}", end="")
                self.pipe_settings[k] = v

    def get_control_net(self, pretrained_path):
        
        print("loading control net")
        # load image processor to annotate images for the control net inputs
        if pretrained_path in ['depth_midas', 'canny', 'openpose', 'normal_bae']:
            self.control_p = Processor(pretrained_path)

        # map custom readable short cuts
        if pretrained_path == 'inpaint':
            pretrained_path = "lllyasviel/control_v11p_sd15_inpaint"
        if pretrained_path == 'depth_midas':
            pretrained_path = "lllyasviel/control_v11f1p_sd15_depth"
        if pretrained_path == 'canny':
            pretrained_path = "lllyasviel/sd-controlnet-canny"
        if pretrained_path == 'openpose':
            pretrained_path = 'lllyasviel/control_v11p_sd15_openpose'
        if pretrained_path == 'normal_bae':
            pretrained_path = 'lllyasviel/control_v11p_sd15_normalbae'

        # Don't load existing model
        if pretrained_path in self.control_nets:
            return self.control_nets[pretrained_path]

        if len(self.control_nets) == 2:
            # release memory of first one
            self.control_nets.pop(0)
        
        self.control_nets[pretrained_path] = ControlNetModel.from_pretrained(pretrained_path, torch_dtype=self.torch_dtype, local_files_only=self.local_files_only)

        return self.control_nets[-1]

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
            self.motion_models[motion_id] = MotionAdapter.from_pretrained(motion_id, torch_dtype=self.torch_dtype, local_files_only=self.local_files_only)
        else:
            self.motion_models[motion_id] = MotionAdapter.from_single_file(motion_id, torch_dtype=self.torch_dtype, local_files_only=self.local_files_only)
        
        return self.motion_models[-1]
    
    def set_pipeline(self):
        if self.pipe_settings.get("mode") == SD15PipelineManager.USE_IMAGE:
            self.pipe = self.get_image_pipe()      
        else:
            self.pipe = self.get_video_pipe()

        # determine if using the inpainting control net
        control_net_id = self.pipe_settings.get("control_net_id", 'openpose')
        if self.pipe_settings.get("use_inpaint_control_net", True):
            # put control nets in list for SD pipeline input
            control_net = [
                self.get_control_net("inpaint"),
                self.get_control_net(control_net_id),
            ]
        else:
            control_net = self.get_control_net(control_net_id)
        self.pipe.control_net = control_net     
        self.pipe.scheduler = self.get_scheduler()
        # # pipeline.to('cuda')
        # load_textual_inversions(pipeline)
        # load_image_loras(pipeline)
        # pipeline.fuse_lora()

        if self.pipe_settings.get("mode") != SD15PipelineManager.USE_IMAGE:
            if self.free_noise_params:
                pipeline.enable_free_noise(context_length=self.free_noise_params[0], context_stride=self.free_noise_params[1])
            if self.free_noise_split_params:
                pipeline.enable_free_noise_split_inference(spatial_split_size=self.free_noise_split_params[0], temporal_split_size=self.free_noise_split_params[1])

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
        if self.torch_dtype == torch.float16 and ('novae' in model_id.lower() or 'no-vae' in model_id.lower() or 'no_vae' in model_id.lower()):
            if self.vae is None:
                if self.local_files_only:
                    print(f"Warning: This model is does not include a VAE and you have local_files_only={self.local_files_only}.")
                self.vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16, local_files_only=self.local_files_only
                )
            pipeline.vae = self.vae
        
        return pipeline
       
    def get_image_pipe(self, model_id="C:\\Users\\teckt\\hf\\models\\single_file\\realisticVisionV60B1_v60B1InpaintingVAE.safetensors"):
        # default can't do anything other than 512x512
        # model_id="SG161222/Realistic_Vision_V5.1_noVAE"
        print("using StableDiffusionControlNetInpaintPipeline")
        if self.pipe_settings.get("use_inpaint_control_net", True):
            pipeline = StableDiffusionControlNetInpaintPipeline.from_single_file(
                model_id, controlnet=None,
                torch_dtype=self.torch_dtype, local_files_only=self.local_files_only,
            ) # requires control image in input
        else:
            pipeline = StableDiffusionInpaintPipeline.from_single_file(
                model_id, controlnet=None,
                torch_dtype=self.torch_dtype, local_files_only=self.local_files_only,
            )

        # auto add vae for some models if vae specified (float16 only)
        if self.torch_dtype == torch.float16 and ('novae' in model_id.lower() or 'no-vae' in model_id.lower() or 'no_vae' in model_id.lower()):
            if self.vae is None:
                if self.local_files_only:
                    print(f"Warning: This model is does not include a VAE and you have local_files_only={self.local_files_only}.")
                self.vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16, local_files_only=self.local_files_only
                )
            pipeline.vae = self.vae

        return pipeline

    def get_scheduler(self):
        schedulers = {
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing='linspace', beta_schedule="linear"),
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

