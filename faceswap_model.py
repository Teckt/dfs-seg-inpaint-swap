import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from numpy._typing import NDArray
from torch import Tensor
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms

### Helper: Pixel Norm
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


### Dense Block (xM option only)
class BuildDense(nn.Module):
    def __init__(self, input_channels, height, width, dims, use_batch_norm=0.0, dropout_ratio=0.0,
                 use_pixel_norm=False, ):
        super().__init__()
        lowest_resolution = height  # as used in the xM branch
        for i in range(5):
            lowest_resolution = lowest_resolution//2
        self.use_pixel_norm = use_pixel_norm
        if input_channels is None:
            raise ValueError("input_channels must be defined.")

        # Use Average Pooling (kernel size 2) to reduce spatial dimensions.
        # self.pool = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        #     nn.Dropout2d(dropout_ratio)
        # )

        fc_input_dim = input_channels
        # Compute flattened dim after pooling:
        # fc_input_dim = input_channels * (height // 2) * (width // 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(fc_input_dim, dims)
        # self.fc_tgt = nn.Linear(dims, dims)
        self.bn1 = nn.BatchNorm1d(dims) if use_batch_norm > 0 else None
        self.fc2 = nn.Linear(dims, lowest_resolution * lowest_resolution * dims)
        self.bn2 = nn.BatchNorm1d(lowest_resolution * lowest_resolution * dims) if use_batch_norm > 0 else None
        self.relu = nn.ReLU(inplace=True)
        self.reshape_size = (dims, lowest_resolution, lowest_resolution)
        self.pixel_norm = PixelNorm() if use_pixel_norm else None

        # self.fc_identity = nn.Linear(dims, dims)
    def forward(self, x):
        # x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)

        # x = self.fc_tgt(x)
        #
        # x = self.relu(x)
        # x = self.fc_identity(x)
        # if self.bn1 is not None:
        #     x = self.bn1(x)
        # x = self.relu(x)
        #
        # x = self.fc_identity(x)
        # if self.bn1 is not None:
        #     x = self.bn1(x)
        # x = self.relu(x)

        if self.pixel_norm:
            x = self.pixel_norm(x)
        x = self.fc2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.relu(x)
        x = x.view(x.size(0), *self.reshape_size)

        return x

class EncoderEff(nn.Module):
    def __init__(self, input_shape):
        """
        input_shape: tuple like (channels, height, width)
        """
        super().__init__()
        self.model = timm.models.create_model('tf_efficientnetv2_b3', pretrained=True)
        # Remove classification head
        self.model.reset_classifier(0)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Forward through encoder
        features = self.model.forward_features(x)  # shape: (B, C, H, W)

        # Global pool to (B, C, 1, 1)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        return pooled

class Encoder(nn.Module):
    def __init__(self, input_shape):
        """
        input_shape: tuple like (channels, height, width)
        """
        super().__init__()
        config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            image_size=input_shape[2]  # or 512, or whatever (must be divisible by 16)
        )
        # Load model with interpolated positional embeddings
        self.model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            config=config,
            ignore_mismatched_sizes=True  # allows positional embedding resizing
        )

    def forward(self, x):
        features = self.model(x)
        features = features.last_hidden_state[:, 0, :]  # Extract CLS token representation
        return features  # use the last feature map


def denormalize(tensor):
    # Undo Normalize(0.5, 0.5): x * 0.5 + 0.5
    tensor = tensor * 0.5 + 0.5
    return tensor


def denormalize_to_pil(tensor):
    # Undo Normalize(0.5, 0.5): x * 0.5 + 0.5
    tensor = tensor * 0.5 + 0.5

    tensor = tensor.clamp(0, 1) * 255  # ensure values are in [0, 1]

    output_transform = transforms.Compose([
        transforms.ToPILImage(),
    ])
    outputs = []
    for i in tensor:
        pil_image = output_transform(i.to(torch.uint8))
        outputs.append(pil_image)
    return outputs


# Full Face Swap Model (xM only)
class FaceSwapModel(nn.Module):
    def __init__(self, input_shape, dims=256, encoder="vit", decoder="pixel_shuffle"):
        """
        Args:
            input_shape: (channels, height, width)
            dims: base dimension multiplier.
        """
        super().__init__()
        self.dims = dims
        self.iterations = 0
        self.encoder_type = encoder
        # Define transformations
        self.transform_training_inputs = \
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_shape[2], input_shape[2]), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=(0, 10), translate=(0., 0.10), scale=(0.9, 1.1), fill=127.5,
                                        interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
        self.transform_training_outputs = denormalize

        # resizes to input and scales to (-1,1)
        self.transform_inference_inputs = \
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_shape[2], input_shape[2]), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
        # PIL Image in 0-255
        self.transform_inference_outputs = denormalize_to_pil

        self.encoder_train_layers = [
            "model.encoder.layer.8",  # 320
            "model.encoder.layer.9",  # 320
            "model.encoder.layer.10",  # 320
            "model.encoder.layer.11",
            "model.layernorm",
            "model.pooler.dense",
        ]
        # self.encoder = EncoderEff(input_shape)
        self.encoder = Encoder(input_shape) if encoder == "vit" else EncoderEff(input_shape)
        # Determine encoder output shape with a dummy tensor.
        dummy = torch.zeros(1, *input_shape)
        with torch.no_grad():
            enc_out = self.encoder(dummy)
        self.enc_shape = enc_out.shape  # (1, C, H, W)
        self.fully_connected = BuildDense(
            input_channels=self.enc_shape[1],
            height=input_shape[1],
            width=input_shape[2],
            dims=dims,
            use_batch_norm=0,
            dropout_ratio=0,
            use_pixel_norm=True
        )

        self.decoder = GenericDecoder(input_channels=dims) if decoder == "pixel_shuffle" else SimpleDecoder(input_channels=dims)

    def load_lora_adapter(self, lora_checkpoint=None, r=64, alpha=1.0):
        """
        loads the lora module to fully_connected.fc2
        if a checkpoint is passed, will use the r and alpha from the checkpoint
        :param lora_checkpoint:
        :param r:
        :return:
        """

        # original weights must be loaded before loading loras

        if isinstance(self.fully_connected.fc2, LoRALinear):
            return

        # switch out modules
        original_state_dict = self.fully_connected.state_dict().copy()

        fc2 = self.fully_connected.fc2

        if lora_checkpoint is not None:
            # load lora weights from checkpoint
            saved_state_dict = torch.load(lora_checkpoint)

            fc2_lora = LoRALinear(
                in_features=fc2.in_features, out_features=fc2.out_features, r=saved_state_dict["r"], alpha=saved_state_dict["alpha"])
        else:
            fc2_lora = LoRALinear(
                in_features=fc2.in_features, out_features=fc2.out_features, r=r, alpha=alpha)
        self.fully_connected.fc2 = fc2_lora

        # switch out state dicts
        lora_state_dict = self.fully_connected.state_dict()

        # transfer original weights to lora module
        lora_state_dict["fc2.linear.weight"] = original_state_dict["fc2.weight"]
        lora_state_dict["fc2.linear.bias"] = original_state_dict["fc2.bias"]

        # load weights from checkpoint
        if lora_checkpoint is not None:
            self.fully_connected.fc2.iterations = saved_state_dict["iterations"]
            lora_state_dict["fc2.A"] = saved_state_dict["A"]
            lora_state_dict["fc2.B"] = saved_state_dict["B"]

        # load everything back to fully connected
        self.fully_connected.load_state_dict(lora_state_dict)

        return

    def save_lora_weights(self, checkpoint_path):
        if not isinstance(self.fully_connected.fc2, LoRALinear):
            return
        fc_state_dict = self.fully_connected.fc2.state_dict()
        keys_to_pop = []
        for k in fc_state_dict.keys():
            if "linear" in k:
                keys_to_pop.append(k)
        for k in keys_to_pop:
            fc_state_dict.pop(k)

        fc_state_dict["r"] = self.fully_connected.fc2.r
        fc_state_dict["alpha"] = self.fully_connected.fc2.alpha
        fc_state_dict["iterations"] = self.fully_connected.fc2.iterations

        torch.save(fc_state_dict, checkpoint_path)

    def prepare_train_lora(self):
        assert isinstance(self.fully_connected.fc2, LoRALinear)

        for i in self.parameters():
            i.requires_grad = False

        for name, param in self.fully_connected.named_parameters():
            if "A" in name or "B" in name:
                param.requires_grad = True

        for name, param in self.fully_connected.named_parameters():
            print(name, param.requires_grad)

        print("all weights frozen except lora weights")

    def forward(self, x):
        enc = self.encoder(x)
        fc = self.fully_connected(enc)
        dec = self.decoder(fc)
        return dec

    def predict(self, x):
        return self.forward(x)

    def load_checkpoint(self, checkpoint_path, fit_decoder_dims=False):
        state_dict = torch.load(
            checkpoint_path
        )

        if "iterations" in state_dict.keys():
            # split dicts
            self.iterations = state_dict["iterations"]

            if fit_decoder_dims:
                model_state_dict = state_dict["model"]
                fitted_state_dict = model_state_dict.copy()
                default_state_dict = self.state_dict()
                for k, param in model_state_dict.items():
                    if "decoder" in k and default_state_dict[k].shape != param.shape:
                        fitted_param = param
                        # reduce by half
                        if default_state_dict[k].shape[0] != param.shape[0]:
                            dim0 = default_state_dict[k].shape[0]
                            fitted_param = fitted_param[:dim0, ...]
                            # fitted_param = F.avg_pool2d(fitted_param, kernel_size=1, stride=2)
                        if len(default_state_dict[k].shape) > 1 and default_state_dict[k].shape[1] != fitted_param.shape[1]:
                            dim1 = default_state_dict[k].shape[1]
                            fitted_param = fitted_param[:, :dim1, ...]
                            # # dynamic shapes compatible
                            # fitted_shape = fitted_param.shape
                            # permute_shape = [*fitted_shape]
                            # # switch dims
                            # permute_shape[0] = fitted_param.shape[1]
                            # permute_shape[1] = fitted_param.shape[0]
                            # fitted_param = torch.permute(fitted_param, permute_shape)
                            # fitted_param = F.avg_pool2d(fitted_param, kernel_size=1, stride=1)
                        fitted_state_dict[k] = fitted_param
                self.load_state_dict(fitted_state_dict, strict=True)
                print("decoder_dims_fitted")
            else:
                self.load_state_dict(state_dict["model"], strict=False)
        else:
            # load as is
            self.load_state_dict(state_dict, strict=False)

    # you can use torch.split or torch.chunk to reduce the number of filters by combining/grouping them
    def combine_filters(self, tensor, num_groups, dim):
        # Split the tensor into groups and return the combined tensors
        chunks = torch.split(tensor, num_groups, dim=dim)
        combined = [torch.cat(group, dim=dim) for group in zip(*chunks)]
        return combined

    def save_checkpoint(self, checkpoint_path):
        state_dict = {
            "model": self.state_dict(),
            "iterations": self.iterations,
        }

        # # split loras and save separately
        # if isinstance(self.fully_connected.fc2, LoRALinear):
        #     model_state_dict = state_dict["model"]
        #     for k in model_state_dict.keys():
        #         if "fully_connected.fc2" in k:


        torch.save(state_dict, checkpoint_path)


    def vit_load_train_layers(self, checkpoint):
        # load saved weights
        state_dict = torch.load(
            checkpoint
        )
        # get all vit weights
        vit_state_dict = self.encoder.state_dict()
        # replace the train layer weights with saved weights
        vit_state_dict.update(state_dict)


        self.encoder.load_state_dict(vit_state_dict)

    def vit_freeze_layers(self):
        # freeze all layers first
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        # unfreeze train layers
        for name, param in self.encoder.named_parameters():
            for train_layer in self.encoder_train_layers:
                if train_layer in name:
                    param.requires_grad = True

    def vit_save_train_layers(self, checkpoint):
        state_dict = {}
        # only save train layers
        for name, param in self.encoder.named_parameters():
            for train_layer in self.encoder_train_layers:
                if train_layer in name:
                    state_dict[name] = param

        torch.save(state_dict, checkpoint)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0):
        super().__init__()
        self.iterations = 0
        self.linear = nn.Linear(in_features, out_features)
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A = nn.Parameter(torch.zeros(r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            self.scaling = alpha / r
        else:
            self.A, self.B = None, None

    def forward(self, x):
        result = self.linear(x)
        if self.r > 0:
            result += (x @ self.A.T @ self.B.T) * self.scaling
        return result


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_batch_norm=False, dropout_ratio=0.0):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.dropout_ratio = dropout_ratio

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding_mode="replicate")
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding_mode="replicate", padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding_mode="replicate")

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return self.relu(x + res)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, pixel_shuffle=True, res_blocks=1):
        super().__init__()
        self.num_res_blocks = res_blocks
        self.conv1 = nn.Conv2d(in_channels, out_channels * 2 * 2, kernel_size=1, stride=1, padding_mode="replicate")
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.use_pixel_shuffle = pixel_shuffle


        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding_mode="replicate")
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding_mode="replicate")

        self.res_block = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)

        # res identity
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        # res
        for _ in range(self.num_res_blocks):
            x = self.res_block(x)

        return x


class GenericDecoder(nn.Module):
    def __init__(self, input_channels, output_channels=3, use_batch_norm=True, dropout_ratio=0.2, res_blocks=2):
        super().__init__()
        decoder_dims = 1280

        self.res_blocks = res_blocks

        self.init_res1 = ResidualBlock(input_channels)
        # self.init_conv2 = nn.Conv2d(input_channels//3, input_channels//3, kernel_size=1, stride=1,
        #                             padding_mode="replicate")
        self.init_conv = nn.Conv2d(input_channels, decoder_dims, kernel_size=3, stride=1,
                                   padding_mode="replicate", padding=1)

        input_channels = decoder_dims
        self.init_res2 = ResidualBlock(input_channels)

        self.up1 = UpsampleBlock(input_channels, input_channels//2, scale_factor=2, res_blocks=res_blocks)  # 7x7 -> 14x14
        self.up2 = UpsampleBlock(input_channels//2, input_channels//4, scale_factor=2, res_blocks=res_blocks)   # 14x14 -> 28x28
        self.up3 = UpsampleBlock(input_channels//4, input_channels//8, scale_factor=2, res_blocks=res_blocks)    # 28x28 -> 56x56
        self.up4 = UpsampleBlock(input_channels//8, input_channels//16, scale_factor=2, res_blocks=res_blocks)    # 56x56 -> 112x112
        self.up5 = UpsampleBlock(input_channels//16, input_channels//32, scale_factor=2, res_blocks=res_blocks)     # 112x112 -> 224x224

        self.final_conv = nn.Conv2d(input_channels//32, output_channels, kernel_size=3, stride=1, padding_mode="replicate", padding=1)
        self.init_res3 = ResidualBlock(3)
        self.final_act = torch.tanh

    def forward(self, x):
        # x = self.init_conv2(x)
        for _ in range(self.res_blocks):
            x = self.init_res1(x)
        x = self.init_conv(x)
        for _ in range(self.res_blocks):
            x = self.init_res2(x)
        # x = self.init_conv2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)

        x = self.final_conv(x)
        for _ in range(self.res_blocks):
            x = self.init_res3(x)

        x = self.final_act(x)  # Output range [-1, 1]
        return x


class SimpleUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResidualBlock(in_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                               padding_mode="replicate")
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                               padding_mode="replicate")

    def forward(self, x):
        x = self.res_block(x)

        x = self.up(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x
class SimpleDecoder(nn.Module):
    def __init__(self, input_channels, output_channels=3, dim_multi=3):
        super().__init__()
        self.init_conv = nn.Conv2d(input_channels, input_channels*dim_multi, kernel_size=1, stride=1,
                                   padding_mode="replicate")
        input_channels = input_channels*dim_multi

        self.up1 = SimpleUpsampleBlock(input_channels, input_channels//2)  # 7x7 -> 14x14
        self.up2 = SimpleUpsampleBlock(input_channels//2, input_channels//4)   # 14x14 -> 28x28
        self.up3 = SimpleUpsampleBlock(input_channels//4, input_channels//8)    # 28x28 -> 56x56
        self.up4 = SimpleUpsampleBlock(input_channels//8, input_channels//16)    # 56x56 -> 112x112
        self.up5 = SimpleUpsampleBlock(input_channels//16, input_channels//32)     # 112x112 -> 224x224
        self.res = ResidualBlock(input_channels//32)

        self.final_conv = nn.Conv2d(input_channels//32, output_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        self.final_act = torch.tanh

    def forward(self, x):
        x = self.init_conv(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.res(x)
        # x = self.res(x)

        x = self.final_conv(x)
        x = self.final_act(x)  # Output range [-1, 1]
        return x


class FaceSwapModel2x(FaceSwapModel):
    def __init__(self, input_shape, dims=256, encoder="vit", decoder="pixel_shuffle"):
        super().__init__(input_shape, dims=dims, encoder=encoder, decoder=decoder)

        # self.fuse_src_tgt = nn.Linear(self.enc_shape[1]*2, self.dims)

        self.decoder_src = GenericDecoder(input_channels=dims) if decoder == "pixel_shuffle" else SimpleDecoder(
            input_channels=dims)

    def forward(self, src, tgt):
        src = self.encoder(src)
        tgt = self.encoder(tgt)

        src = self.fully_connected(src)
        tgt = self.fully_connected(tgt)

        # fused = torch.cat([src, tgt], dim=1)
        # fused = self.fuse_src_tgt(fused)
        #
        # fused = self.fully_connected(fused)
        # 3) Decode

        tgt = self.decoder(tgt)
        src = self.decoder_src(src)
        return src, tgt

    def predict(self, src):
        src = self.encoder(src)
        src = self.fully_connected(src)
        src = self.decoder(src)
        return src

class FS3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
class FS3Dense(nn.Module):
    def __init__(self, input_channels, height, width, dims):
        super().__init__()
        lowest_resolution = height  # as used in the xM branch
        for i in range(5):
            lowest_resolution = lowest_resolution // 2

        if input_channels is None:
            raise ValueError("input_channels must be defined.")

        fc_input_dim = input_channels
        # Compute flattened dim after pooling:
        # fc_input_dim = input_channels * (height // 2) * (width // 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(fc_input_dim, dims)

        self.fc2 = nn.Linear(dims, lowest_resolution * lowest_resolution * dims)

        self.relu = nn.ReLU(inplace=True)
        self.reshape_size = (dims, lowest_resolution, lowest_resolution)
        self.pixel_norm = PixelNorm()

        # self.fc_identity = nn.Linear(dims, dims)

    def forward(self, x):

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.pixel_norm(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = x.view(x.size(0), *self.reshape_size)

        return x
class FS3Decoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.init_conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1,
                                    padding_mode="replicate", padding=1)

        self.up1 = UpsampleBlock(input_channels, input_channels//2, scale_factor=2, res_blocks=4)  # 7x7 -> 14x14
        self.up2 = UpsampleBlock(input_channels//2, input_channels//4, scale_factor=2, res_blocks=4)   # 14x14 -> 28x28
        self.up3 = UpsampleBlock(input_channels//4, input_channels//8, scale_factor=2, res_blocks=4)    # 28x28 -> 56x56
        self.up4 = UpsampleBlock(input_channels//8, input_channels//16, scale_factor=2, res_blocks=4)    # 56x56 -> 112x112
        self.up5 = UpsampleBlock(input_channels//16, input_channels//32, scale_factor=2, res_blocks=4)     # 112x112 -> 224x224

        self.final_conv = nn.Conv2d(input_channels//32, 3, kernel_size=3, stride=1, padding_mode="replicate", padding=1)
        self.final_act = torch.tanh

    def forward(self, x):
        # x = self.init_conv2(x)
        x = self.init_conv(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)

        x = self.final_conv(x)
        x = self.final_act(x)  # Output range [-1, 1]
        return x

class FS3(FaceSwapModel):
    def __init__(self, input_shape, dims=1024, encoder="vit", decoder="pixel_shuffle"):
        super().__init__(input_shape, dims=dims, encoder=encoder, decoder=decoder)

        # self.fuse_src_tgt = nn.Linear(self.enc_shape[1]*2, self.dims)

        self.encoder = FS3Encoder()
        self.decoder = GenericDecoder(dims)

    def forward(self, src, tgt):
        src = self.encoder(src)
        tgt = self.encoder(tgt)



        src = self.fully_connected(src)
        tgt = self.fully_connected(tgt)

        fused = self.fuse_src_tgt(torch.cat([src * 0.2, tgt * 0.8], dim=1))
        # fused = torch.cat([src, tgt], dim=1)
        # fused = self.fuse_src_tgt(fused)
        #
        # fused = self.fully_connected(fused)
        # 3) Decode

        tgt = self.decoder(tgt)
        # src = self.decoder_src(src)
        return src, tgt

    def predict(self, src):
        src = self.encoder(src)
        src = self.fully_connected(src)
        src = self.decoder(src)
        return src


### Example Usage
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = FaceSwapModel((3, 224, 224), dims=512).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    y = fs(x)

    fs.load_lora_adapter(r=128, alpha=math.sqrt(128))
    fs.prepare_train_lora()
    fs.save_lora_weights("faceswap/fc2_lora.pth")

