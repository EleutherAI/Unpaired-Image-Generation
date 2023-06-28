import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2

cap = dset.CocoCaptions(root = 'coco/images/train2014',
                        annFile = 'coco/annotations/annotations_trainval2014/annotations/captions_train2014.json',
                        transform=transforms.PILToTensor())

print('Number of samples: ', len(cap))

for img, target in cap:
    viewable_img = img.permute(1, 2, 0).numpy()
    # rgb to bgr
    viewable_img = viewable_img[:, :, ::-1].copy()
    viewable_img = cv2.putText(viewable_img, target[0], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('img', viewable_img)
    cv2.waitKey(0)
    print("Image Size: ", img.size())
    print("target: ", target)