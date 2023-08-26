import torch
import torch.nn as nn

from transformers import T5Tokenizer, T5ForConditionalGeneration
from model.pl_resnet_ae import get_resnet18_encoder
from tqdm import tqdm

# very crude implementation of unpaired t2i and i2t with diffusion
# TODO:
    # cross attention for mixing latents
    # sampling from latent space
    # masking for text and image for unpaired data
    # better sampling for noise steps
    # integrate with coco dataset


class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(Encoder, self).__init__()
        t5_size = 'small'
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-' + t5_size)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-' + t5_size)

        self.img_encoder = get_resnet18_encoder(first_conv=False, maxpool1=False)

        self.img_proj = nn.LazyLinear(latent_dim)
        self.text_proj = nn.LazyLinear(latent_dim)

        # replace with cross attention later
        self.mixer = nn.LazyLinear(latent_dim)

    def forward(self, text, image):
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        text_encoder_out = self.t5.encoder(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
        text_feats = text_encoder_out.last_hidden_state

        img_feats = self.img_encoder(image)

        # flatten the image features
        img_feats = img_feats.view(img_feats.size(0), -1)
        img_feats = self.img_proj(img_feats)

        # flatten the text features
        text_feats = text_feats.view(text_feats.size(0), -1)
        text_feats = self.text_proj(text_feats)

        # mix the features
        latent = self.mixer(img_feats + text_feats)

        # add sampling from the latent space

        return text_feats, img_feats, latent

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # scale U-Net 

        # 3x224x224 to 64x56x56
        self.inp = nn.Sequential(
            nn.Conv2d(3, 64, 3, 4, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 64x56x56 to 2x16x16
        self.down = nn.Sequential(
            nn.Conv2d(64, 2, 4, 4, 4),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )

        # 2x16x16 to 64x56x56
        self.up = nn.Sequential(
            nn.ConvTranspose2d(2, 64, 4, 4, 4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 64x56x56 to 3x224x224
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 4, 0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

    def forward(self, image, latent):
        out = self.inp(image)
        out = self.down(out)

        out = out.flatten()
        latent = latent.flatten()

        # need to combine the latent space with timestep encoding and the image features
        print('out:', out.shape, 'latent:', latent.shape)
        out = out + latent
        # reshape ,512 to ,2,16,16
        out = out.view(-1, 2, 16, 16)

        out = self.up(out)
        out = self.out(out)

        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.t5_size = 'small'
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-' + self.t5_size)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-' + self.t5_size)
        self.text_decoder_proj = nn.LazyLinear(512)

    def forward(self, latent):
        # pass the latent space through a linear layer to get the text decoder input
        text_decoder_input = self.text_decoder_proj(latent)
        txt_out = self.tokenizer.batch_decode(text_decoder_input, skip_special_tokens=True)

        return txt_out, text_decoder_input


# sourced from
# https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=224):
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
    
    def noise_images_batched(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
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
        # model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x

def train(img, text):
    enc = Encoder()
    dec_i = UNet()
    dec_t = Decoder()
    diffuse = Diffusion()

    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec_i.parameters()) + list(dec_t.parameters()), lr=1e-4)
    mse = nn.MSELoss()

    for i in range(10):
        # better sampling for the noise steps
        noise_steps = torch.randint(0, 1000, (20,))
        print(noise_steps)

        text_feats, img_feats, latent = enc(text, img)
        _, text_tensor = dec_t(latent)

        loss = mse(text_tensor, text_feats)
        for ns in noise_steps:
            noised_img, noise = diffuse.noise_images(img, ns)
            pred_noise = dec_i(noised_img, latent)

            loss += mse(pred_noise, noise)

        print(torch.sum(noised_img - pred_noise))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(loss.item()) if i % 100 == 0 else None
        print(loss.item())

if __name__ == "__main__":
    img = torch.randn(8, 3, 224, 224)
    text = ["A dog is running in the park" for _ in range(8)]

    train(img, text)
