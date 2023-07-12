import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
from model.t2ivae import T2IVAE
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import argparse
import numpy as np

def train_epoch(model, train_loader, optimizer):
    model.train()
    loss_sum = 0
    for i, (img, text) in enumerate(tqdm(train_loader)):
        # print('text: ', text)
        img = img.to(device).float()
        # text_input = model.tokenizer(text, return_tensors="pt", padding=True).to(device)
        text_input = text.to(device)
        # text_input = model.tokenizer.batch_encode_plus(
        #             text[0],
        #             padding='max_length',
        #             max_length=32,
        #             return_tensors='pt',
        #             truncation=True
        #             )["input_ids"].to(device)
        print('tokenized text shape: ', text_input["input_ids"].shape) # token ids

        optimizer.zero_grad()

        output = model(img, text_input)

        if args.debug and i % 50 == 0:
            visualize_data(img, text_input, output)

        loss = criterion(output, img, text_input)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
    
    return loss_sum / len(train_loader)

def val_epoch(model, val_loader):
    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for i, (img, text) in enumerate(tqdm(val_loader)):
            # print('text: ', text)
            img = img.to(device).float()
            # text_input = model.tokenizer(text, return_tensors="pt", padding=True).to(device)
            text_input = text.to(device)
            print('tokenized text: ', text_input) # dict 
            print('tokenized text shape: ', text_input["input_ids"].shape) # token ids
            # text_input = model.tokenizer.batch_encode_plus(
            #             text[0],
            #             padding='max_length',
            #             max_length=32,
            #             return_tensors='pt',
            #             truncation=True
            #             )["input_ids"].to(device)

            output = model(img, text_input)

            if args.debug and i % 50 == 0:
                visualize_data(img, text_input, output)

            loss = criterion(output, img, text_input)
            loss_sum += loss.item()
    
    return loss_sum / len(val_loader)

def kl_divergence(mu1, logvar1, mu2, logvar2):
    # kl divergence from means and logvars
    # # kl_div = (q_log_var-p_log_var + (jnp.exp(p_log_var)+(p_mean-q_mean)**2)/jnp.exp(q_log_var)-1)/2 

    # sanity check
    # var1 = torch.exp(logvar1)
    # var2 = torch.exp(logvar2)
    # kl_div = 0.5 * (torch.log(var2 / var1) + (var1 + (mu1 - mu2)**2) / (var2) - 1)
    # print('kl test: ', kl_div.sum())

    # Calculate KL divergence
    kl_div = 0.5 * (logvar2 - logvar1 + (torch.exp(logvar1) + (mu1 - mu2)**2) / torch.exp(logvar2) - 1) # (batch_size, hidden_dim)

    return torch.mean(kl_div) # summing over all elements in the batch

def get_text_loss(pred, target):
    # pred: (batch_size, seq_len, vocab_size)
    # target: (batch_size, seq_len)
    pred = pred.view(-1, pred.size(-1)) # (batch_size*seq_len, vocab_size)
    target = target.view(-1) # (batch_size*seq_len)

    print('pred: ', pred, 'target: ', target)
    print('pred size: ', pred.size(), 'target size: ', target.size())

    # cross entropy loss, ignoring padding
    loss = torch.nn.functional.cross_entropy(pred, target, ignore_index=0)
    print('text loss size: ', loss.size())

    return loss

def criterion(output, img, text_input):
    # applying L1 loss between output['pred_img] and img
    img_loss = torch.nn.functional.l1_loss(output['pred_img'], img) # TODO: change to NLL loss
    text_loss = get_text_loss(output['pred_text'], text_input['input_ids'])

    # applying unit gaussian prior to image features
    img_prior_mean = torch.zeros_like(output['img_feat_means']).to(device)
    img_prior_logvar = torch.zeros_like(output['img_feat_logvars']).to(device)
    img_kl_loss = kl_divergence(output['img_feat_means'], output['img_feat_logvars'], img_prior_mean, img_prior_logvar)

    # applying unit gaussian prior to text features
    text_prior_mean = torch.zeros_like(output['text_feat_means']).to(device)
    text_prior_logvar = torch.zeros_like(output['text_feat_logvars']).to(device)
    text_kl_loss = kl_divergence(output['text_feat_means'], output['text_feat_logvars'], text_prior_mean, text_prior_logvar)

    # kl divergence between image and text features
    img_text_kl_loss = kl_divergence(output['img_feat_means'], output['img_feat_logvars'], output['text_feat_means'], output['text_feat_logvars'])
    
    if args.debug:
        print('img_loss: ', img_loss)
        print('text_loss: ', text_loss)
        print('img_kl_loss: ', img_kl_loss)
        print('text_kl_loss: ', text_kl_loss)
        print('img_text_kl_loss: ', img_text_kl_loss)

    # return img_loss + text_loss  # + img_kl_loss + text_kl_loss + img_text_kl_loss
    if img is None and text_input is not None:
        return text_loss + text_kl_loss
    elif img is not None and text_input is None:
        return img_loss + img_kl_loss
    else:
        return 5 * img_loss + 10 * text_loss + 0.001 * img_kl_loss + 0.001 * text_kl_loss + 0.01 * img_text_kl_loss

def get_viewable_text(token_ids):
    # decoding text from token ids, stopping at eos token
    viewable_text = ''

    # # getting index of eos token
    # eos_token_idx = token_ids.tolist().index(1)

    # checking if eos token is in token_ids
    if 1 in token_ids:
        eos_token_idx = token_ids.tolist().index(1)
    else:
        eos_token_idx = len(token_ids) - 1

    # decoding text
    viewable_text = model.tokenizer.decode(token_ids[:eos_token_idx])

    return viewable_text

def tensor_to_cv2(tensor):
    # input: tensor of shape (3, 224, 224)
    # output: numpy array of shape (224, 224, 3)
    if tensor.requires_grad:
        img = tensor.permute(1, 2, 0).cpu().detach().numpy()
    else:
        img = tensor.permute(1, 2, 0).cpu().numpy()
    # rgb to bgr
    img = img[:, :, ::-1].copy()
    # resizing to 4x
    img = cv2.resize(img, (0, 0), fx=4, fy=4)
    return img


def visualize_data(img_input, text_input, output=None):
    # visualize the data given the inputs or outputs
    # img: [batch_size, 3, 224, 224]
    # text_input: [batch_size, max_seq_len]
    # output: [batch_size, 3, 224, 224]
    # output: [batch_size, max_seq_len, vocab_size]

    # visualizing image and caption
    gt_img = tensor_to_cv2(img_input[0])

    # decoding text from token ids
    # viewable_text = model.tokenizer.decode(text_input["input_ids"][0], skip_special_tokens=True)
    viewable_text_gt = get_viewable_text(text_input["input_ids"][0])

    gt_img = cv2.putText(gt_img, 'ground truth: ' + viewable_text_gt, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.imshow('input img', gt_img)

    if output is not None:
        # visualizing predicted image and caption
        img_from_img = tensor_to_cv2(output['pred_img'][0])

        # getting predicted token ids
        pred_token_ids = torch.argmax(output['pred_text'][0], dim=1)
        viewable_text_pred = get_viewable_text(pred_token_ids)

        img_from_img = cv2.putText(img_from_img, 'VAE: ' + viewable_text_pred, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.imshow('pred_img_from_img', img_from_img)

        img_from_text = tensor_to_cv2(output['pred_img_t2i'][0])
        img_from_text = cv2.putText(img_from_text, 't2i prompt: ' + viewable_text_gt, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.imshow('pred_img_t2i', img_from_text)
        disp_img = np.concatenate((gt_img, img_from_img, img_from_text), axis=1)
    else:
        disp_img = gt_img

    cv2.imshow('disp_img', disp_img)
    cv2.waitKey(1)

def custom_collate_fn(batch):
    images, texts = zip(*batch)

    # getting just the first caption from each element in the batch
    # and prepending each caption with "summarize: "
    # texts = ["summarize: " + text[0] for text in texts]

    texts = [text[0] for text in texts]

    # text_input = model.tokenizer(texts[0], return_tensors="pt", padding=True) # ["input_ids"]

    # setting max_seq_len to 128
    text_input = model.tokenizer(texts, return_tensors="pt", padding=True, max_length=model.max_seq_len, truncation=True) # ["input_ids"]

    # # Define the image transformations
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    # ])

    # Apply transformations to each image in the batch
    # images = [transform(image) for image in images]

    # Convert images list into a PyTorch tensor
    images = torch.stack(images)

    print('text_input ids: ', text_input["input_ids"].shape)
    # Pad sequences for text
    if text_input["input_ids"].size(1) < model.max_seq_len:
        text_input["input_ids"] = torch.nn.functional.pad(text_input["input_ids"], (0, model.max_seq_len - text_input["input_ids"].shape[1]))
    else:
        text_input["input_ids"] = text_input["input_ids"][:, :model.max_seq_len] # truncate to max seq len

    # setting attention mask
    # text_input["attention_mask"] = torch.ones(text_input["input_ids"].shape)
    # ignoring padding tokens
    text_input["attention_mask"] = (text_input["input_ids"] != model.tokenizer.pad_token_id)

    print('padded text_input ids: ', text_input["input_ids"].shape)

    # so we can access the raw text later
    # text_input["raw_text"] = torch.tensor(texts)

    return images, text_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T2IVAE().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers)
    
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer)
        val_epoch(model, val_loader)
        print("Epoch: ", epoch)

        # saving model
        torch.save(model.state_dict(), 'checkpoints/t2i_vae.pt')
        print("Saved model")

    print('Number of samples: ', len(train_loader))


    
        