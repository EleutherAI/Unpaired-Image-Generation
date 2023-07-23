import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms
import cv2
from model.t2ivae import T2IVAE
from tqdm import tqdm
import argparse
import numpy as np
from model.utils import *
from model.config_utils import parse_config_args
import random
import wandb
import datetime
import os

def train_epoch(model, train_loader, optimizer):
    model.train()
    loss_sum = 0
    avg_loss_dict = {}
    for i, (img, text) in enumerate(tqdm(train_loader)):
        if hasattr(config, 'WARMUP_EPOCHS') and epoch < config.WARMUP_EPOCHS:
            print('in warmup phase')
            mask_img, mask_text = False, False
        elif config.MASKING:
            print('in masking phase')
            mask_img, mask_text = get_masks() 
        else:
            mask_img, mask_text = False, False

        # print('text: ', text)
        img = img.to(device).float()
        # text_input = model.tokenizer(text, return_tensors="pt", padding=True).to(device)
        text_input = text.to(device)          

        optimizer.zero_grad()

        output = model(img, text_input, mask_img, mask_text)

        if args.debug and i % 10 == 0:
            disp_img = visualize_data(img, text_input, model.tokenizer, output, config, mask_img, mask_text)
            cv2.imshow('disp_img', disp_img)
            cv2.waitKey(1)
            datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # cv2.imwrite('logs/' + args.config + '/train_' + str(epoch) + '_' + datetime_str + '.jpg', (disp_img * 255).astype(np.uint8))

        loss_dict = criterion(output, img, text_input, mask_img, mask_text)
        loss = loss_dict['loss_total']
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        # for key in loss_dict:
        #     wandb.log({key + '_train': loss_dict[key].item()})

        # recording the running average of the loss
        for key in loss_dict:
            if key not in avg_loss_dict: # first time
                avg_loss_dict[key] = loss_dict[key].item()
            else:
                avg_loss_dict[key] += loss_dict[key].item()

    for key in avg_loss_dict:
        if key == 'loss_total':
            avg_loss_dict[key] /= len(train_loader)
        else:
            avg_loss_dict[key] /= (len(train_loader) / 3) # each loss only occurs 1/3 of the time
        
        wandb.log({key + '_avg_train': avg_loss_dict[key]}, step=epoch)

    return loss_sum / len(train_loader)

def val_epoch(model, val_loader):
    model.eval()
    loss_sum = 0
    avg_loss_dict = {}
    with torch.no_grad():
        for i, (img, text) in enumerate(tqdm(val_loader)):
            if config.MASKING:
                mask_img, mask_text = get_masks() 
            else:
                mask_img, mask_text = False, False
            
            # print('text: ', text)
            img = img.to(device).float()
            # text_input = model.tokenizer(text, return_tensors="pt", padding=True).to(device)
            text_input = text.to(device)

            output = model(img, text_input, mask_img, mask_text)

            if args.debug and i % 10 == 0:
                disp_img = visualize_data(img, text_input, model.tokenizer, output, config, mask_img, mask_text)
                cv2.imshow('disp_img', disp_img)
                cv2.waitKey(1)
                datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                # cv2.imwrite('logs/val_' + str(epoch) + '_' + datetime_str + '.jpg', (disp_img * 255).astype(np.uint8))
                cv2.imwrite('logs/' + args.config + '/val_' + str(epoch) + '_' + datetime_str + '.jpg', (disp_img * 255).astype(np.uint8))


            loss_dict = criterion(output, img, text_input, mask_img, mask_text)
            loss = loss_dict['loss_total']
            loss_sum += loss.item()

            # for key in loss_dict:
            #     wandb.log({key + '_val': loss_dict[key].item()})

            # recording the running average of the loss
            for key in loss_dict:
                if key not in avg_loss_dict:
                    avg_loss_dict[key] = loss_dict[key].item()
                else:
                    avg_loss_dict[key] += loss_dict[key].item()

    for key in avg_loss_dict:
        if key == 'loss_total':
            avg_loss_dict[key] /= len(val_loader)
        else:
            avg_loss_dict[key] /= (len(val_loader) / 3) # each loss only occurs 1/3 of the time
        
        wandb.log({key + '_avg_val': avg_loss_dict[key]}, step=epoch)

    return loss_sum / len(val_loader)

def get_text_loss(pred, target):
    # pred: (batch_size, seq_len, vocab_size)
    # target: (batch_size, seq_len)
    pred = pred.view(-1, pred.size(-1)) # (batch_size*seq_len, vocab_size)
    target = target.view(-1) # (batch_size*seq_len)

    # print('pred size: ', pred.size(), 'target size: ', target.size())
    # cross entropy loss, ignoring padding
    loss = torch.nn.functional.cross_entropy(pred, target, ignore_index=0)

    return loss

def criterion(output, img, text_input, mask_img, mask_text):
    # applying L1 loss between output['pred_img] and img
    # img_loss = torch.nn.functional.l1_loss(output['pred_img'], img) # TODO: change to gaussian NLL loss
    img_loss = torch.nn.GaussianNLLLoss()(output['pred_img_means'], img, torch.exp(output['pred_img_logvars'])) # input, target, variance
    text_loss = get_text_loss(output['pred_text'], text_input['input_ids'])

    # applying unit gaussian prior to combined image and text features
    combined_prior_mean = torch.zeros_like(output['combined_embedding_means']).to(device)
    combined_prior_logvar = torch.zeros_like(output['combined_embedding_logvars']).to(device)
    combined_kl_loss = kl_divergence(output['combined_embedding_means'], output['combined_embedding_logvars'], combined_prior_mean, combined_prior_logvar)

    # kl divergence between image and text features
    # img_text_kl_loss = kl_divergence(output['img_feat_means'], output['img_feat_logvars'], output['text_feat_means'], output['text_feat_logvars'])

    if args.debug:
        print('img_loss: ', img_loss)
        print('text_loss: ', text_loss)
        print('kl_loss: ', combined_kl_loss)        

    if mask_img:
        # loss_total = config.LAMBDA_TEXT * text_loss + config.LAMBDA_KL * combined_kl_loss
        loss_total = config.LAMBDA_IMAGE * img_loss + config.LAMBDA_KL * combined_kl_loss
        # loss_total = config.LAMBDA_IMAGE * img_loss + config.LAMBDA_TEXT * text_loss + config.LAMBDA_KL * combined_kl_loss
        
        return {
            'loss_total': loss_total,
            'text_only_loss_total': loss_total,
            't2t_loss': text_loss,
            't2i_loss': img_loss,
            'text_only_combined_kl_loss': combined_kl_loss
        }
    elif mask_text:
        # loss_total = config.LAMBDA_IMAGE * img_loss + config.LAMBDA_KL * combined_kl_loss
        loss_total = config.LAMBDA_TEXT * text_loss + config.LAMBDA_KL * combined_kl_loss
        # loss_total = config.LAMBDA_IMAGE * img_loss + config.LAMBDA_TEXT * text_loss + config.LAMBDA_KL * combined_kl_loss
        return {
            'loss_total': loss_total,
            'img_only_loss_total': loss_total,
            'i2i_loss': img_loss,
            'i2t_loss': text_loss,
            'img_only_combined_kl_loss': combined_kl_loss
        }
    else:
        loss_total = config.LAMBDA_IMAGE * img_loss + config.LAMBDA_TEXT * text_loss + config.LAMBDA_KL * combined_kl_loss

        return {
                'loss_total': loss_total,
                'combined_loss_total': loss_total,
                'combined_img_loss': img_loss,
                'combined_text_loss': text_loss,
                'combined_kl_loss': combined_kl_loss
                }

def custom_collate_fn(batch):
    images, texts = zip(*batch)

    if config.DATASET == 'coco':
        # generating a random caption index from 0 to 4 for each image
        texts = [text[random.randint(0, len(text) - 1)] for text in texts]
    elif config.DATASET == 'cifar100':
        # turning the CIFAR 100 class index into a string
        texts = [config.CIFAR100_CLASSES[text] for text in texts]

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

    # so we can access the raw text later
    # text_input["raw_text"] = torch.tensor(texts)

    return images, text_input

if __name__ == "__main__":
    config, args = parse_config_args()
    wandb.init(project='unpaired_t2i')
    wandb.config.update(config)
    wandb.config.update(args)
    
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.run.name = args.config + '_' + datetime_str

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('logs/' + args.config):
        os.makedirs('logs/' + args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T2IVAE().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if hasattr(config, 'LR_SCHEDULER') and config.LR_SCHEDULE:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_SCHEDULE_STEP, gamma=config.LR_SCHEDULE_GAMMA) # multiply lr by gamma every step_size epochs
    else:
        scheduler = None

    if config.DATASET == 'coco':
        train_dataset = dset.CocoCaptions(root = 'coco/images/train2014',
                                annFile = 'coco/annotations/annotations_trainval2014/annotations/captions_train2014.json',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                ]))
        
        val_dataset = dset.CocoCaptions(root = 'coco/images/val2014',
                                annFile = 'coco/annotations/annotations_trainval2014/annotations/captions_val2014.json',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224))
                                ]))
    elif config.DATASET == 'cifar100':
        train_dataset = dset.CIFAR100(root='cifar100', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((32, 32))
                            ]))

        val_dataset = dset.CIFAR100(root='cifar100', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((32, 32))
                            ]))

        # train_dataset = MaskedDataset(dataset_class=dset.CIFAR100, model=model, root='cifar100', train=True, download=True,
        #                     transform=transforms.Compose([
        #                         transforms.ToTensor(),
        #                         transforms.Resize((32, 32))
        #                     ]))

        # val_dataset = MaskedDataset(dataset_class=dset.CIFAR100, model=model, root='cifar100', train=False, download=True,
        #                     transform=transforms.Compose([
        #                         transforms.ToTensor(),
        #                         transforms.Resize((32, 32))
        #                     ]))
        
        
        # loading the cifar class names from a text file
        with open('cifar100_labels.txt', 'r') as f:
            config.CIFAR100_CLASSES = f.read().splitlines()
            print('class names:', config.CIFAR100_CLASSES)
    else:
        print('Dataset not supported')
        raise NotImplementedError

    # masking out half of the images and half of the text
    # # train_dataset is a list of tuples (image, caption)
    # for i in range(len(train_dataset)):
    #     mask_img = random.randint(0, 1)
    #     mask_text = random.randint(0, 1)
    #     if mask_img:
    #         print('debug img', train_dataset[i][0])
    #         train_dataset[i][0] = torch.zeros_like(list(train_dataset[i])[0])
    #     if mask_text:
    #         print('debug text', train_dataset[i][1])
    #         # setting input ids to just model.tokenizer.eos_token_id
    #         train_dataset[i][1]["input_ids"] = torch.full_like(list(train_dataset[i])[1]["input_ids"], model.tokenizer.eos_token_id)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    # train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS, sampler=SubsetRandomSampler(range(13))) # for debugging
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    # val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS, sampler=SubsetRandomSampler(range(13))) # for debugging

    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = val_epoch(model, val_loader)
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)
        print("Epoch: ", epoch)

        # saving model
        torch.save(model.state_dict(), 'checkpoints/' + args.config + '_latest.pt')
        print("Saved latest model")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/' + args.config + '_best.pt')
            print("Saved best model")
        if epoch % 100 == 0:
            torch.save(model.state_dict(), 'checkpoints/' + args.config + '_epoch' + str(epoch) + '.pt')
            print("Saved model at epoch ", epoch)

    print('Number of samples: ', len(train_loader))


    
        