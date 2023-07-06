import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
from model.t2ivae import T2IVAE
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import argparse

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
    kl_div = 0.5 * (logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mu1 - mu2)**2 / torch.exp(logvar2) - 1)
    # # kl_div = (q_log_var-p_log_var + (jnp.exp(p_log_var)+(p_mean-q_mean)**2)/jnp.exp(q_log_var)-1)/2
    # kl_div = (logvar1 - logvar2 + (torch.exp(logvar1) + (mu1 - mu2)**2)/torch.exp(logvar2) - 1)/2
    # print('kl_div size: ', kl_div.shape)
    return torch.mean(kl_div) # summing over all elements in the batch

def criterion(output, img, text_input):
    # applying L1 loss between output['pred_img] and img
    print('img size: ', img.size(), 'pred_img size: ', output['pred_img'].size())
    img_loss = torch.nn.functional.l1_loss(output['pred_img'], img)
    print('img_loss size: ', img_loss.size())

    # applying cross-entropy loss between output['pred_text'] and text_input['input_ids']
    # padding
    print('pred_text size: ', output['pred_text'].shape, 'input_ids size: ', text_input['input_ids'].shape)
    # text_loss = torch.nn.functional.cross_entropy(output['pred_text'], text_input['input_ids'])

    pred_text_logits_flat = output['pred_text'].view(-1, output['pred_text'].size(-1))  # shape: [8*26, 32128]
    input_ids_flat = text_input['input_ids'].view(-1)  # shape: [8*26]

    print('pred_text size: ', pred_text_logits_flat.size(), 'input_ids size: ', input_ids_flat.size())
    # text_loss = torch.nn.functional.cross_entropy(pred_text_logits_flat, input_ids_flat) # TODO: replace with MLM objective
    # print('text_loss size: ', text_loss.size())
    text_loss = 0


    # applying unit gaussian prior to image features
    img_prior_mean = torch.zeros_like(output['img_feat_means']).to(device)
    img_prior_logvar = torch.zeros_like(output['img_feat_logvars']).to(device)
    img_kl_loss = kl_divergence(output['img_feat_means'], output['img_feat_logvars'], img_prior_mean, img_prior_logvar)

    # applying unit gaussian prior to text features
    text_prior_mean = torch.zeros_like(output['text_feat_means']).to(device)
    text_prior_logvar = torch.zeros_like(output['text_feat_logvars']).to(device)
    text_kl_loss = kl_divergence(output['text_feat_means'], output['text_feat_logvars'], text_prior_mean, text_prior_logvar)

    # applying KL divergence loss between output['img_feat_means'], output['img_feat_logvars'], output['text_feat_means'], output['text_feat_logvars']
    print('img_feat_means size: ', output['img_feat_means'].size(), 'img_feat_logvars size: ', output['img_feat_logvars'].size())
    print('text_feat_means size: ', output['text_feat_means'].size(), 'text_feat_logvars size: ', output['text_feat_logvars'].size())
    # kl divergence from means and logvars
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
        return img_loss + text_loss + 0.01 * img_kl_loss # + text_kl_loss + img_text_kl_loss

def visualize_data(img_input, text_input, output=None):
    # visualize the data given the inputs or outputs
    # img: [batch_size, 3, 224, 224]
    # text_input: [batch_size, max_seq_len]
    # output: [batch_size, 3, 224, 224]
    # output: [batch_size, max_seq_len, vocab_size]

    # # visualizing image and caption
    viewable_img = img_input[0].permute(1, 2, 0).cpu().numpy()
    # rgb to bgr
    viewable_img = viewable_img[:, :, ::-1].copy()

    # resizing to 4x
    viewable_img = cv2.resize(viewable_img, (0, 0), fx=4, fy=4)

    # decoding text from token ids
    viewable_text = model.tokenizer.decode(text_input["input_ids"][0])
    viewable_img = cv2.putText(viewable_img, viewable_text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('input img', viewable_img)

    if output is not None:
        # visualizing predicted image and caption
        viewable_img = output['pred_img'][0].permute(1, 2, 0).cpu().detach().numpy()
        # rgb to bgr
        viewable_img = viewable_img[:, :, ::-1].copy()

        # resizing to 4x
        viewable_img = cv2.resize(viewable_img, (0, 0), fx=4, fy=4)

        # decoding text from token ids
        # getting predicted token ids
        pred_token_ids = torch.argmax(output['pred_text'][0], dim=1)
        # getting rid of padding token ids
        pred_token_ids = pred_token_ids[pred_token_ids != 0]
        viewable_text = model.tokenizer.decode(pred_token_ids)
        viewable_img = cv2.putText(viewable_img, viewable_text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('pred_img', viewable_img)

    cv2.waitKey(1)

def custom_collate_fn(batch):
    images, texts = zip(*batch)

    # getting just the first caption from each element in the batch
    # and prepending each caption with "summarize: "
    texts = ["summarize: " + text[0] for text in texts]
    text_input = model.tokenizer(texts[0], return_tensors="pt", padding=True) # ["input_ids"]

    # # Define the image transformations
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    # ])

    # Apply transformations to each image in the batch
    # images = [transform(image) for image in images]

    # Convert images list into a PyTorch tensor
    images = torch.stack(images)

    print('text_input ids: ', text_input["input_ids"])
    # Pad sequences for text
    text_input["input_ids"] = pad_sequence([torch.tensor(t) for t in text_input["input_ids"]], batch_first=True)
    print('padded text_input ids: ', text_input["input_ids"])

    # so we can access the raw text later
    # text_input["raw_text"] = torch.tensor(texts)

    return images, text_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T2IVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train_loader = dset.CocoCaptions(root = 'coco/images/train2014',
    #                         annFile = 'coco/annotations/annotations_trainval2014/annotations/captions_train2014.json',
    #                         transform=transforms.PILToTensor(),)

    # changing training transforms to include resizing
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

    print('Number of samples: ', len(train_loader))


    
        