import torch
import torch.nn as nn

from transformers import T5Tokenizer, T5ForConditionalGeneration
from model.pl_resnet_ae import get_resnet18_encoder
from tqdm import tqdm
import cv2

'''
Crude implementation of unpaired t2i and i2t with diffusion.
Currently, we are only using the Diffusion class from this file.
'''
# TODO:
    # cross attention for mixing latents
    # better sampling for noise steps
    # integrate with coco dataset

# sourced from
# https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py
class Diffusion:
    def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, img_size=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = (1. - self.beta).to(self.device)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device)

        self.img_size = img_size

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        print('x:', x.shape)
        print('t:', t.shape)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        print('sqrt_alpha_hat:', sqrt_alpha_hat.shape)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).view(-1, 1, 1, 1)
        print('sqrt_one_minus_alpha_hat:', sqrt_one_minus_alpha_hat.shape)
        Ɛ = torch.randn_like(x)
        # noised image, noise
        print('sqrt_alpha_hat * x', (sqrt_alpha_hat * x).shape)
        print('sqrt_one_minus_alpha_hat * Ɛ', (sqrt_one_minus_alpha_hat * Ɛ).shape)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n, labels, cfg_scale=3):
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # print('i:', i)
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * -1 * predicted_noise) + torch.sqrt(beta) * noise

                # display the image
                img = (x.clamp(-1, 1) + 1) / 2
                img = (img * 255).type(torch.uint8)
                img = img[0].permute(1, 2, 0).cpu().numpy()
                # rgb to bgr
                img = img[:, :, ::-1].copy()
                img = cv2.resize(img, (0, 0), fx=28, fy=28, interpolation=cv2.INTER_NEAREST)
                cv2.imshow('disp_img', img)
                cv2.waitKey(1)
        
        # model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x