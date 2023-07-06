import torch
import torch.nn as nn
import timm
from transformers import T5Model, T5EncoderModel, T5Tokenizer, BertModel, BertTokenizer, T5ForConditionalGeneration
import numpy as np

class T2IVAE(nn.Module):
    def __init__(self):
        super(T2IVAE, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.img_encoder = timm.create_model('vit_small_patch16_224', pretrained=False, num_c).to(device) # hidden_size: 768
        self.img_encoder = timm.create_model('resnet18', pretrained=False, num_classes=0).to(device) # hidden_size: 512
        self.img_encoder.global_pool = nn.Identity() # so we can get the feature map
        self.img_decoder = self.get_img_decoder().to(device)
        t5_size = 'small'
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-' + t5_size).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-' + t5_size)

        self.latent_size = 1024

        self.img_feat_means = nn.LazyLinear(self.latent_size)
        self.img_feat_logvars = nn.LazyLinear(self.latent_size)

        self.text_feat_means = nn.LazyLinear(self.latent_size)
        self.text_feat_logvars = nn.LazyLinear(self.latent_size)

        # linear projection to 512 * 7 * 7
        self.deconv_proj = nn.LazyLinear(512 * 7 * 7)

    def get_img_decoder(self):
        # convolutional decoder # TODO: replace with diffusion decoder
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 512, 7, 7 -> 256, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 256, 14, 14 -> 128, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 128, 28, 28 -> 64, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 64, 56, 56 -> 32, 112, 112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), # 32, 112, 112 -> 3, 224, 224
            nn.Sigmoid() # pixel intensity should be between 0 and 1
        )
    
    def sample_gaussian(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img, text_inputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text = text_inputs["input_ids"].to(device) # token ids (batch_size, seq_len)

        img_feats = self.img_encoder(img)
        # pred_img = self.img_decoder(img_feats)
        # pred_text = self.t5.generate(input_ids=text, attention_mask=text_inputs["attention_mask"].to(device))
        # Modify this line to get the output from the T5 model directly
        # output_t5 = self.t5(input_ids=text, attention_mask=text_inputs["attention_mask"].to(device), decoder_input_ids=text[:, :-1])
        output_t5 = self.t5(input_ids=text, attention_mask=text_inputs["attention_mask"].to(device), decoder_input_ids=text)

        # Get the logits from the T5 model's output
        pred_text = output_t5.logits
        print("pred_text in forward: ", pred_text.shape) # (batch_size, seq_len, vocab_size)

        # padding each sequence if it is less than 100 tokens, else truncate it
        if pred_text.shape[1] < 100:
            pred_text = torch.nn.functional.pad(pred_text, (0, 0, 0, 100 - pred_text.shape[1]))
        else:
            pred_text = pred_text[:, :100, :]

        print("pred_text in forward after padding: ", pred_text.shape) # (batch_size, max_len, vocab_size)
        pred_text = output_t5.logits.view(-1, text.shape[1], self.t5.config.vocab_size)

        # getting the hidden state from the encoder
        text_feats = self.t5.get_encoder()(input_ids=text, attention_mask=text_inputs["attention_mask"].to(device)).last_hidden_state
        print("text_feats: ", text_feats.shape) # (batch_size, seq_len, hidden_size)

        # padding each sequence if it is less than 100 tokens, else truncate it
        if text_feats.shape[1] < 100:
            text_feats = torch.nn.functional.pad(text_feats, (0, 0, 0, 100 - text_feats.shape[1]))
        else:
            text_feats = text_feats[:, :100, :]
        print("text_feats after padding: ", text_feats.shape) # (batch_size, max_len, hidden_size)

        flattened_img_feats = img_feats.view(img_feats.shape[0], -1).to(device)
        flattened_text_feats = text_feats.view(text_feats.shape[0], -1).to(device)

        # print("flattened_img_feats: ", flattened_img_feats.shape)
        # print("flattened_text_feats: ", flattened_text_feats.shape)

        # proj_img_feats = self.img_projection(flattened_img_feats)
        # proj_text_feats = self.text_projection(flattened_text_feats)

        img_feat_means = self.img_feat_means(flattened_img_feats)
        img_feat_logvars = self.img_feat_logvars(flattened_img_feats)
        sampled_latent = self.sample_gaussian(img_feat_means, img_feat_logvars)
        img_decoder_input = self.deconv_proj(sampled_latent).view(-1, 512, 7, 7)
        pred_img = self.img_decoder(img_decoder_input)

        text_feat_means = self.text_feat_means(flattened_text_feats)
        text_feat_logvars = self.text_feat_logvars(flattened_text_feats)
        text_decoder_input = self.sample_gaussian(text_feat_means, text_feat_logvars)

        return {
            "pred_img": pred_img,
            "pred_text": pred_text,
            "img_feats": img_feats,
            "text_feats": text_feats,
            "img_feat_means": img_feat_means,
            "img_feat_logvars": img_feat_logvars,
            "text_feat_means": text_feat_means,
            "text_feat_logvars": text_feat_logvars
        }
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T2IVAE().to(device)
    img = torch.randn(1, 3, 224, 224).to(device)
    text = "summarize: This is a sentence."
    text = model.tokenizer(text, return_tensors="pt", padding=True)
    output = model(img, text)

    print("pred_img", output["pred_img"].shape)
    print("pred_text", output["pred_text"].shape)
    print("img_feats", output["img_feats"].shape)
    print("text_feats: ", output["text_feats"].shape)
    # print("proj_img_feats: ", output["proj_img_feats"].shape)
    # print("proj_text_feats: ", output["proj_text_feats"].shape)
    print("img_feat_means: ", output["img_feat_means"].shape)
    print("img_feat_logvars: ", output["img_feat_logvars"].shape)
    print("text_feat_means: ", output["text_feat_means"].shape)
    print("text_feat_logvars: ", output["text_feat_logvars"].shape)

    decoded_text = model.tokenizer.batch_decode(output["pred_text"], skip_special_tokens=True)
    print("decoded pred_text: ", decoded_text)