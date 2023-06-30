import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
from model.t2ivae import T2IVAE
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

def train_epoch(model, train_loader, optimizer):
    model.train()
    loss_sum = 0
    for img, text in tqdm(train_loader):
        print('text: ', text)
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

        optimizer.zero_grad()

        output = model(img, text_input)

        loss = criterion(output, img, text_input)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
    
    return loss_sum / len(train_loader)

def val_epoch(model, val_loader):
    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for img, text in tqdm(train_loader):
            print('text: ', text)
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

            loss = criterion(output, img, text_input)
            loss_sum += loss.item()
    
    return loss_sum / len(val_loader)

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
    text_loss = torch.nn.functional.cross_entropy(pred_text_logits_flat, input_ids_flat)
    print('text_loss size: ', text_loss.size())
    # text_loss = 0

    # applying KL divergence loss between output['img_feat_means'], output['img_feat_logvars'], output['text_feat_means'], output['text_feat_logvars']
    print('img_feat_means size: ', output['img_feat_means'].size(), 'img_feat_logvars size: ', output['img_feat_logvars'].size())
    print('text_feat_means size: ', output['text_feat_means'].size(), 'text_feat_logvars size: ', output['text_feat_logvars'].size())
    # kl divergence from means and logvars
    # img_kl_loss = -0.5 * torch.sum(1 + output['img_feat_logvars'] - output['img_feat_means'].pow(2) - output['img_feat_logvars'].exp())
    # kl_div = (q_log_var-p_log_var + (jnp.exp(p_log_var)+(p_mean-q_mean)**2)/jnp.exp(q_log_var)-1)/2
    kl_div = torch.sum((output['img_feat_logvars'] - output['text_feat_logvars'] + (torch.exp(output['text_feat_logvars']) + (output['text_feat_means'] - output['img_feat_means'])**2)/torch.exp(output['img_feat_logvars']) - 1)/2) # TODO: check this
    
    print('kl_div size: ', kl_div.size())

    return img_loss + text_loss + kl_div

def custom_collate_fn(batch):
    images, texts = zip(*batch)

    # getting just the first caption from each element in the batch
    # and prepending each caption with "summarize: "
    texts = ["summarize: " + text[0] for text in texts]
    print('texts in collate_fn: ', texts)

    text_input = model.tokenizer(texts[0], return_tensors="pt", padding=True) # ["input_ids"]
    print('text_input in collate_fn: ', text_input["input_ids"].shape)

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

    return images, text_input

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T2IVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
                            transform=transforms.PILToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    
    for epoch in range(10):
        train_epoch(model, train_loader, optimizer)
        val_epoch(model, val_loader)
        print("Epoch: ", epoch)

    print('Number of samples: ', len(train_loader))


    
        