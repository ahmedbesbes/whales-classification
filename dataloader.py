import pandas as pd
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import expand2square
import albumentations as A
import albumentations.pytorch as AT


class WhalesData(Dataset):
    def __init__(self, paths, bbox, mapping_label_id, transform, crop=False, test=False):
        self.paths = paths
        self.bbox = pd.read_csv(bbox)
        self.bbox.set_index('new_path', inplace=True)
        self.bbox = self.bbox.to_dict(orient='index')
        self.mapping_label_id = mapping_label_id
        self.transform = transform
        self.test = test
        self.crop = crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]

        # img = io.imread(path)
        img = Image.open(path)
        img = np.array(img)
        if self.crop:
            if (path in self.bbox) & ('test' not in path):
                x = int(self.bbox[path]['x'])
                y = int(self.bbox[path]['y'])
                w = int(self.bbox[path]['w'])
                h = int(self.bbox[path]['h'])
            else:
                x, y = 0, 0
                w = img.shape[1]
                h = img.shape[0]
            img = img[y:h, x:w, :]

        if type(self.transform) == A.core.composition.Compose:
            img = self.transform(image=img)['image']
        else:
            img = self.transform(img)

        if self.test == False:
            folder = path.split('/')[-2]
            label = self.mapping_label_id[folder]
            sample = {
                'image': img,
                'label': label
            }
        else:
            sample = {
                'image': img
            }
        return sample


def augmentation(image_size, train=True, heavy=False):
    max_crop = image_size // 10
    if train:
        if heavy:
            data_transform = A.Compose([
                A.OneOf([
                    A.RandomRain(),
                    A.GaussNoise(mean=25),
                    A.GaussianBlur(blur_limit=20),
                    A.MotionBlur(10)
                ]),

                A.OneOf([
                    A.RGBShift(p=1.0, r_shift_limit=(-5, 5),
                               g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.1, p=1),
                    A.HueSaturationValue(hue_shift_limit=20, p=1),
                ]),

                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.IAASharpen(),
                    A.IAAEmboss(),
                ],
                    p=0.1),

                A.IAAPerspective(p=0.3),

                A.IAAAffine(scale=0.9,
                            translate_px=15,
                            rotate=20,
                            shear=0.2,
                            p=1),

                #A.Cutout(num_holes=1, max_h_size=100, max_w_size=200, p=0.2),

                A.Resize(image_size, image_size),
                A.Cutout(num_holes=1, max_h_size=max_crop,
                         max_w_size=max_crop, p=0.2),

                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                AT.ToTensor()
            ])
        else:
            data_transform = A.Compose([
                A.OneOf([
                    A.GaussNoise(mean=10),
                    A.GaussianBlur(blur_limit=10)
                ]),
                A.IAAPerspective(scale=(0.1, 0.01)),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1),
                A.HueSaturationValue(hue_shift_limit=10),
                A.IAAAffine(scale=1, translate_px=10, rotate=10, shear=1),
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                AT.ToTensor()
            ])

    else:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return data_transform
