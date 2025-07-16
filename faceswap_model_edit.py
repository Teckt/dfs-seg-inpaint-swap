from dfs_seg_inpaint_swap.faceswap_model import FaceSwapModel
import torch
if __name__ == "__main__":
    checkpoint_path_320 = f"faceswap/autoencoder_{320}_{512}.pth"
    enc_path = f"faceswap/autoencoder_224_512.pth-enc.pth"
    dec_path = f"faceswap/autoencoder_224_512.pth-dec.pth"

    # model_320 = FaceSwapModel((3, 320, 320), 512, encoder="", decoder="")
    # model_320.load_checkpoint(checkpoint_path_320)

    model_224 = FaceSwapModel((3, 224, 224), 512, encoder="")
    model_224.encoder.load_state_dict(torch.load(enc_path))
    model_224.decoder.load_state_dict(torch.load(dec_path))

    model_224.save_checkpoint(f"faceswap/autoencoder_{224}_{512}.pth")