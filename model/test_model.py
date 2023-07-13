import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
from model.t2ivae import T2IVAE
from tqdm import tqdm
import argparse
import numpy as np
from model.utils import *
from model.trainer import custom_collate_fn
from model.config_utils import parse_config_args

stage = 'val'

config, args = parse_config_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T2IVAE().to(device)
model.load_state_dict(torch.load('checkpoints/t2i_vae.pt'))
model.eval()

def custom_collate_fn(batch):
    images, texts = zip(*batch)
    texts = [text[0] for text in texts]

    text_input = model.tokenizer(texts, return_tensors="pt", padding=True, max_length=config.MAX_SEQ_LEN, truncation=True) # ["input_ids"]

    # Convert images list into a PyTorch tensor
    images = torch.stack(images)

    # Pad sequences for text
    if text_input["input_ids"].size(1) < config.MAX_SEQ_LEN:
        text_input["input_ids"] = torch.nn.functional.pad(text_input["input_ids"], (0, config.MAX_SEQ_LEN - text_input["input_ids"].shape[1]))
    else:
        text_input["input_ids"] = text_input["input_ids"][:, :config.MAX_SEQ_LEN] # truncate to max seq len

    # ignoring padding tokens
    text_input["attention_mask"] = (text_input["input_ids"] != model.tokenizer.pad_token_id)

    return images, text_input


if stage == 'train':
    dataset = dset.CocoCaptions(root = 'coco/images/train2014',
                                annFile = 'coco/annotations/annotations_trainval2014/annotations/captions_train2014.json',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                ]))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
    
elif stage == 'val':
    dataset = dset.CocoCaptions(root = 'coco/images/val2014',
                                annFile = 'coco/annotations/annotations_trainval2014/annotations/captions_val2014.json',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224))
                                ]))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)

with torch.no_grad():
    for i, (img, text) in enumerate(tqdm(loader)):
        img = img.to(device).float()
        text_input = text.to(device)

        output = model(img, text_input)
        disp_img = visualize_data(img, text_input, model.tokenizer, output)
        
        cv2.imshow('img', disp_img)
        cv2.waitKey(0)