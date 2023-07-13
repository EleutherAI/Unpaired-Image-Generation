import torch
import torch.nn as nn
import timm
from transformers import T5Model, T5EncoderModel, T5Tokenizer, BertModel, BertTokenizer, T5ForConditionalGeneration
import numpy as np
from model.config_utils import parse_config_args

class T2IVAE(nn.Module):
    def __init__(self):
        super(T2IVAE, self).__init__()
        self.config, args = parse_config_args()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.img_encoder = timm.create_model('vit_small_patch16_224', pretrained=False, num_c).to(device) # hidden_size: 768
        self.img_encoder = timm.create_model('resnet18', pretrained=False, num_classes=0).to(device) # hidden_size: 512
        self.img_encoder.global_pool = nn.Identity() # so we can get the feature map
        self.img_decoder = self.get_img_decoder().to(device)
        t5_size = 'small'
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-' + t5_size).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-' + t5_size)

        self.img_feat_means = nn.LazyLinear(self.config.LATENT_DIM)
        self.img_feat_logvars = nn.LazyLinear(self.config.LATENT_DIM)

        self.text_feat_means = nn.LazyLinear(self.config.LATENT_DIM)
        self.text_feat_logvars = nn.LazyLinear(self.config.LATENT_DIM)

        # linear projection to 512 * 7 * 7
        self.img_decoder_proj = nn.LazyLinear(512 * 7 * 7)
        self.text_decoder_proj = nn.LazyLinear(512 * self.config.MAX_SEQ_LEN)

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

        # pred_text = self.t5.generate(input_ids=text, attention_mask=text_inputs["attention_mask"].to(device))
        
        # creating placeholder for text decoder
        # create placeholder input with start token at the first position and pad tokens elsewhere
        # self.placeholder_input = torch.full((self.config.BATCH_SIZE, self.config.MAX_SEQ_LEN), self.tokenizer.pad_token_id).to(device)
        self.placeholder_input = torch.full((text.shape[0], self.config.MAX_SEQ_LEN), self.tokenizer.pad_token_id).to(device) # to address end of batch error

        self.placeholder_input[:, 0] = self.tokenizer.eos_token_id # set first token to be start token

        # text encoder
        text_encoder_out = self.t5.encoder(input_ids=text, attention_mask=text_inputs["attention_mask"].to(device))
        text_feats = text_encoder_out.last_hidden_state

        flattened_img_feats = img_feats.view(img_feats.shape[0], -1).to(device)
        flattened_text_feats = text_feats.view(text_feats.shape[0], -1).to(device)

        img_feat_means = self.img_feat_means(flattened_img_feats)
        img_feat_logvars = self.img_feat_logvars(flattened_img_feats) - 4 # lowering logvars by subtracting a constant (e.g. 3-5ish)
        sampled_img_latent = self.sample_gaussian(img_feat_means, img_feat_logvars)
        img_decoder_input = self.img_decoder_proj(sampled_img_latent).view(-1, 512, 7, 7)
        pred_img = self.img_decoder(img_decoder_input)

        text_feat_means = self.text_feat_means(flattened_text_feats)
        text_feat_logvars = self.text_feat_logvars(flattened_text_feats) - 4 # lowering logvars by subtracting a constant (e.g. 3-5ish)
        sampled_text_latent = self.sample_gaussian(text_feat_means, text_feat_logvars)
        text_decoder_input = self.text_decoder_proj(sampled_text_latent).view(-1, self.config.MAX_SEQ_LEN, 512)

        print('text_decoder_input: ', text_decoder_input.shape)
        print('text_feats: ', text_feats.shape)

        # text decoder
        # text_decoder_out = self.t5.decoder(input_ids=text, encoder_hidden_states=text_feats, encoder_attention_mask=text_inputs["attention_mask"].to(device))
        text_decoder_out = self.t5.decoder(input_ids=self.placeholder_input, encoder_hidden_states=text_decoder_input, encoder_attention_mask=text_inputs["attention_mask"].to(device))
        pred_text = self.t5.lm_head(text_decoder_out.last_hidden_state) # (batch_size, seq_len, vocab_size)
        print("pred_text in forward: ", pred_text.shape)
        pred_text = pred_text.view(-1, text.shape[1], self.t5.config.vocab_size)


        # text to image
        t2i_input = self.img_decoder_proj(sampled_text_latent).view(-1, 512, 7, 7)
        pred_img_t2i = self.img_decoder(t2i_input)

        # image to text
        i2t_input = self.text_decoder_proj(sampled_img_latent).view(-1, self.config.MAX_SEQ_LEN, 512)
        # i2t_decoder_out = self.t5.decoder(input_ids=text, encoder_hidden_states=i2t_input, encoder_attention_mask=text_inputs["attention_mask"].to(device))
        i2t_decoder_out = self.t5.decoder(input_ids=self.placeholder_input, encoder_hidden_states=i2t_input, encoder_attention_mask=text_inputs["attention_mask"].to(device))
        pred_text_i2t = self.t5.lm_head(i2t_decoder_out.last_hidden_state) # (batch_size, seq_len, vocab_size)
        pred_text_i2t = pred_text_i2t.view(-1, text.shape[1], self.t5.config.vocab_size)

        return {
            "pred_img": pred_img,
            "pred_text": pred_text,
            "img_feats": img_feats,
            "text_feats": text_feats,
            "img_feat_means": img_feat_means,
            "img_feat_logvars": img_feat_logvars,
            "text_feat_means": text_feat_means,
            "text_feat_logvars": text_feat_logvars,
            "pred_img_t2i": pred_img_t2i,
            "pred_text_i2t": pred_text_i2t
        }
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T2IVAE().to(device)
    img = torch.randn(1, 3, 224, 224).to(device)
    text = "This is a sentence."
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