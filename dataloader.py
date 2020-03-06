import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from utils import expand2square


class WhalesData(Dataset):
    def __init__(self, paths, bbox, mapping_label_id, transform, test=False):
        self.paths = paths
        self.bbox = pd.read_csv(bbox)
        self.bbox.set_index('path', inplace=True)
        self.bbox = self.bbox.to_dict(orient='index')
        self.mapping_label_id = mapping_label_id
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]

        img = io.imread(path)
        if path in self.bbox:
            x = int(self.bbox[path]['x'])
            y = int(self.bbox[path]['y'])
            w = int(self.bbox[path]['w'])
            h = int(self.bbox[path]['h'])
        else:
            x, y = 0, 0
            w = img.shape[1]
            h = img.shape[0]
        img = img[y:h, x:w, :]
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


def augmentation(image_size, train=True):
    if train:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda img: expand2square(img)),
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda img: expand2square(img)),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    return data_transform
