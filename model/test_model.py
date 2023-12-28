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
import random

stage = 'val'
print('stage:', stage)

config, args = parse_config_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T2IVAE().to(device)
model.load_state_dict(torch.load('checkpoints/' + args.model + '.pt'))
model.eval()

def custom_collate_fn(batch):
    images, texts = zip(*batch)

    if config.DATASET == 'coco':
        # generating a random caption index from 0 to 4 for each image
        texts = [text[random.randint(0, len(text) - 1)] for text in texts]
    elif config.DATASET == 'cifar100':
        # turning the CIFAR 100 class index into a string
        texts = [config.CIFAR100_CLASSES[text] for text in texts]
    elif config.DATASET == 'cifar10':
        # turning the CIFAR 10 class index into a string
        texts = [config.CIFAR10_CLASSES[text] for text in texts]

    text_input = model.tokenizer(texts, return_tensors="pt", padding=True, max_length=config.MAX_SEQ_LEN, truncation=True) # ["input_ids"]

    # # Define the image transformations
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    # ])

    # Apply transformations to each image in the batch
    # images = [transform(image) for image in images]

    # Convert images list into a PyTorch tensor
    images = torch.stack(images)

    # Pad sequences for text
    if text_input["input_ids"].size(1) < config.MAX_SEQ_LEN:
        text_input["input_ids"] = torch.nn.functional.pad(text_input["input_ids"], (0, config.MAX_SEQ_LEN - text_input["input_ids"].shape[1]))
    else:
        text_input["input_ids"] = text_input["input_ids"][:, :config.MAX_SEQ_LEN] # truncate to max seq len

    # setting attention mask
    # ignoring padding tokens
    text_input["attention_mask"] = (text_input["input_ids"] != model.tokenizer.pad_token_id)

    print('padded text_input ids: ', text_input["input_ids"].shape)

    # so we can access the raw text later
    # text_input["raw_text"] = torch.tensor(texts)

    return images, text_input

if config.DATASET == 'coco':
    if stage == 'train':
        dataset = dset.CocoCaptions(root = 'coco/images/train2014',
                                annFile = 'coco/annotations/annotations_trainval2014/annotations/captions_train2014.json',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                ]))
    elif stage == 'val':
        dataset = dset.CocoCaptions(root = 'coco/images/val2014',
                                annFile = 'coco/annotations/annotations_trainval2014/annotations/captions_val2014.json',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224))
                                ]))
        
elif config.DATASET == 'cifar100':
    if stage == 'train':
        dataset = dset.CIFAR100(root='cifar100', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((32, 32))
                            ]))
    elif stage == 'val':
        dataset = dset.CIFAR100(root='cifar100', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((32, 32))
                            ]))
    
    # loading the cifar class names from a text file
    with open('cifar100_labels.txt', 'r') as f:
        config.CIFAR100_CLASSES = f.read().splitlines()
        print('class names:', config.CIFAR100_CLASSES)
        
elif config.DATASET == 'cifar10':
    if stage == 'train':
        dataset = dset.CIFAR10(root='cifar10', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((32, 32))
                            ]))
    elif stage == 'val':
        dataset = dset.CIFAR10(root='cifar10', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((32, 32))
                            ]))
    
    # loading the cifar class names from a text file
    with open('cifar10_labels.txt', 'r') as f:
        config.CIFAR10_CLASSES = f.read().splitlines()
        print('class names:', config.CIFAR10_CLASSES)

loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

with torch.no_grad():
    for i, (img, text) in enumerate(tqdm(loader)):
        img = img.to(device).float()
        text_input = text.to(device)

        if config.MASKING:
            if args.t2i:
                mask_img, mask_text = True, False
            elif args.i2t:
                mask_img, mask_text = False, True
            else:
                mask_img, mask_text = get_masks() 
        else:
            mask_img, mask_text = False, False

        output = model(img, text_input, mask_img, mask_text)
        print('combined embedding means (mean, std): ', output['combined_embedding_means'].mean(), output['combined_embedding_means'].std())
        print('combined embedding logvars (mean, std): ', output['combined_embedding_logvars'].mean(), output['combined_embedding_logvars'].std())
        disp_img = visualize_data(img, text_input, model.tokenizer, output, config, mask_img, mask_text, model, sample_diffusion=True)
        
        cv2.imshow('img', disp_img)
        cv2.waitKey(0)