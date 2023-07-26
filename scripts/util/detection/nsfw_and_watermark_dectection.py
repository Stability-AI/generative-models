import os

import clip
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

RESOURCES_ROOT = "scripts/util/detection/"


def predict_proba(X, weights, biases):
    logits = X @ weights.T + biases
    proba = np.where(
        logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits))
    )
    return proba.T


def load_model_weights(path: str):
    model_weights = np.load(path)
    return model_weights["weights"], model_weights["biases"]


def clip_process_images(images: torch.Tensor) -> torch.Tensor:
    min_size = min(images.shape[-2:])
    return T.Compose(
        [
            T.CenterCrop(min_size),  # TODO: this might affect the watermark, check this
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )(images)


class DeepFloydDataFiltering(object):
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.clip_model, _ = clip.load("ViT-L/14", device="cpu")
        self.clip_model.eval()

        self.cpu_w_weights, self.cpu_w_biases = load_model_weights(
            os.path.join(RESOURCES_ROOT, "w_head_v1.npz")
        )
        self.cpu_p_weights, self.cpu_p_biases = load_model_weights(
            os.path.join(RESOURCES_ROOT, "p_head_v1.npz")
        )
        self.w_threshold, self.p_threshold = 0.5, 0.5

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        imgs = clip_process_images(images)
        image_features = self.clip_model.encode_image(imgs.to("cpu"))
        image_features = image_features.detach().cpu().numpy().astype(np.float16)
        p_pred = predict_proba(image_features, self.cpu_p_weights, self.cpu_p_biases)
        w_pred = predict_proba(image_features, self.cpu_w_weights, self.cpu_w_biases)
        print(f"p_pred = {p_pred}, w_pred = {w_pred}") if self.verbose else None
        query = p_pred > self.p_threshold
        if query.sum() > 0:
            print(f"Hit for p_threshold: {p_pred}") if self.verbose else None
            images[query] = T.GaussianBlur(99, sigma=(100.0, 100.0))(images[query])
        query = w_pred > self.w_threshold
        if query.sum() > 0:
            print(f"Hit for w_threshold: {w_pred}") if self.verbose else None
            images[query] = T.GaussianBlur(99, sigma=(100.0, 100.0))(images[query])
        return images


def load_img(path: str) -> torch.Tensor:
    image = Image.open(path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image_transforms = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    return image_transforms(image)[None, ...]


def test(root):
    from einops import rearrange

    filter = DeepFloydDataFiltering(verbose=True)
    for p in os.listdir((root)):
        print(f"running on {p}...")
        img = load_img(os.path.join(root, p))
        filtered_img = filter(img)
        filtered_img = rearrange(
            255.0 * (filtered_img.numpy())[0], "c h w -> h w c"
        ).astype(np.uint8)
        Image.fromarray(filtered_img).save(
            os.path.join(root, f"{os.path.splitext(p)[0]}-filtered.jpg")
        )


if __name__ == "__main__":
    import fire

    fire.Fire(test)
    print("done.")
