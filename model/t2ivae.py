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
        t5_size = 'large'
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-' + t5_size).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-' + t5_size)

        self.img_projection = nn.LazyLinear(1024)
        self.text_projection = nn.LazyLinear(1024)

    def get_img_decoder(self):
        # convolutional decoder
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

    def forward(self, img, text_inputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text = text_inputs["input_ids"].to(device)
        print("img: ", img.shape)
        print("text tokens: ", text.shape) # tokenized text

        img_feats = self.img_encoder(img)
        pred_img = self.img_decoder(img_feats)
        pred_text = self.t5.generate(input_ids=text, attention_mask=text_inputs["attention_mask"].to(device))
        
        # getting the hidden state from the encoder
        text_feats = self.t5.get_encoder()(input_ids=text, attention_mask=text_inputs["attention_mask"].to(device)).last_hidden_state

        flattened_img_feats = img_feats.view(img_feats.shape[0], -1).to(device)
        flattened_text_feats = text_feats.view(text_feats.shape[0], -1).to(device)

        # print("flattened_img_feats: ", flattened_img_feats.shape)
        # print("flattened_text_feats: ", flattened_text_feats.shape)

        proj_img_feats = self.img_projection(flattened_img_feats)
        proj_text_feats = self.text_projection(flattened_text_feats)

        return {
            "pred_img": pred_img,
            "pred_text": pred_text,
            "img_feats": img_feats,
            "text_feats": text_feats,
            "proj_img_feats": proj_img_feats,
            "proj_text_feats": proj_text_feats,
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
    print("proj_img_feats: ", output["proj_img_feats"].shape)
    print("proj_text_feats: ", output["proj_text_feats"].shape)
    
    decoded_text = model.tokenizer.batch_decode(output["pred_text"], skip_special_tokens=True)
    print("decoded pred_text: ", decoded_text)