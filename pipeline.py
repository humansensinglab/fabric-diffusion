import numpy as np
import torch
import torch.nn as nn
import os
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image


class FabricDiffusionPipeline():
    def __init__(self, device, texture_checkpoint, print_checkpoint):
        self.device = device
        self.texture_checkpoint = texture_checkpoint
        self.print_base_model = print_checkpoint

        if texture_checkpoint:
            self.texture_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                texture_checkpoint, 
                torch_dtype=torch.float16,
                safety_checker=None
            )
            # with open(os.path.join(texture_checkpoint, "unet", "diffusion_pytorch_model.safetensors"), "rb") as f:
            #     data = f.read()
            # loaded = load(data)
            # self.texture_pipeline.unet.load_state_dict(loaded)
            self.texture_model = self.texture_model.to(device)
        else:
            self.texture_model = None

        # set circular convolution for the texture model
        if self.texture_model:
            for a, b in self.texture_model.unet.named_modules():
                if isinstance(b, nn.Conv2d):
                    setattr(b, 'padding_mode', 'circular')
            for a, b in self.texture_model.vae.named_modules():
                if isinstance(b, nn.Conv2d):
                    setattr(b, 'padding_mode', 'circular')

        if print_checkpoint:
            self.print_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                print_checkpoint, 
                torch_dtype=torch.float16,
                safety_checker=None
            )
            self.print_model = self.print_model.to(device)
        else:
            self.print_model = None

    def load_real_data_with_mask(self, dataset_path, image_name):
        image = np.array(Image.open(os.path.join(dataset_path, 'images', image_name)).convert('RGB'))
        seg_mask = np.array(Image.open(os.path.join(dataset_path, 'seg_mask', image_name)).convert('L'))[..., None]
        texture_mask = np.array(Image.open(os.path.join(dataset_path, 'texture_mask', image_name)).convert('L'))[..., None]
        # crop the image based on texture_mask
        x1, y1, x2, y2 = np.where(texture_mask > 0)[1].min(), np.where(texture_mask > 0)[0].min(), np.where(texture_mask > 0)[1].max(), np.where(texture_mask > 0)[0].max()
        texture_patch = image[y1:y2, x1:x2]
        # resize the texture_patch to 256x256
        texture_patch = Image.fromarray(texture_patch.astype(np.uint8)).resize((256, 256))

        return image, seg_mask, texture_patch
    
    def load_patch_data(self, patch_path):
        texture_patch = Image.open(patch_path).convert('RGB').resize((256, 256))
        return texture_patch

    def flatten_texture(self, texture_patch, n_samples=3, use_inversion=True):
        num_inference_steps = 20
        self.texture_model.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.texture_model.scheduler.timesteps

        # convert image to latent using vae
        image = self.texture_model.image_processor.preprocess(texture_patch)
        if use_inversion:
            image_latents = self.texture_model.prepare_image_latents(image, batch_size=1, 
                                                          num_images_per_prompt=1,
                                                          device=self.device,
                                                          dtype=torch.float16,
                                                          do_classifier_free_guidance=False)
            
            image_latents = (image_latents - torch.mean(image_latents)) / torch.std(image_latents)

            # forward noising process
            noise = torch.randn_like(image_latents)
            noisy_image_latents = self.texture_model.scheduler.add_noise(image_latents, noise, timesteps[0:1])
            noisy_image_latents /= self.texture_model.scheduler.init_noise_sigma
            noisy_image_latents = torch.tile(noisy_image_latents, (n_samples, 1, 1, 1))
        else:
            noisy_image_latents = None

        image = torch.tile(image, (n_samples, 1, 1, 1))
        gen_imgs = self.texture_model(
            "",
            image=image,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7.,
            latents=noisy_image_latents,
            num_images_per_prompt=n_samples,
        ).images

        return gen_imgs

    def flatten_print(self, print_patch, n_samples=3):
        image = self.print_model.image_processor.preprocess(print_patch)
        gen_imgs = []
        for i in range(n_samples):
            gen_img = self.print_model(
                "",
                image=image,
                num_inference_steps=20,
                image_guidance_scale=1.5,
                guidance_scale=7.,
            ).images[0]
            gen_img = np.asarray(gen_img) / 255.
            alpha_map = np.clip(gen_img / 0.1 * 1.2 - 0.2, 0., 1).mean(axis=-1, keepdims=True)
            gen_img = np.clip((gen_img - 0.1) / 0.9, 0., 1.)
            gen_img = np.concatenate([gen_img, alpha_map], axis=-1)
            gen_img = (gen_img * 255).astype(np.uint8)
            gen_img = Image.fromarray(gen_img)
            gen_imgs.append(gen_img)

        return gen_imgs

