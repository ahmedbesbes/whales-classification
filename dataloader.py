import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import expand2square


class WhalesDataset(Dataset):
    def __init__(self, num_triplets, transform, root, stats_classes, mapping_class_to_images, ids_to_labels, bbox):

        self.num_triplets = num_triplets
        self.root = root
        self.classes = os.listdir(self.root)
        self.stats_classes = stats_classes
        self.mapping_class_to_images = mapping_class_to_images
        self.training_triplets = self.create_triplets()
        self.transform = transform
        self.ids_to_labels = ids_to_labels
        self.bbox = pd.read_csv(bbox)
        self.bbox.set_index('path', inplace=True)
        self.bbox = self.bbox.to_dict(orient='index')

    def create_triplets(self):
        triplets = []

        for _ in tqdm(range(self.num_triplets), leave=False):
            random_pos_class = (self.stats_classes[self.stats_classes['num'] > 1]['class']
                                .sample(1)
                                .values[0])

            random_neg_class = (self.stats_classes[self.stats_classes['class'] != random_pos_class]['class']
                                .sample(1)
                                .values[0])

            anc_file, pos_file = np.random.choice(
                self.mapping_class_to_images[random_pos_class], 2, replace=False)
            neg_file = np.random.choice(
                self.mapping_class_to_images[random_neg_class])

            anc_file = os.path.join(self.root, random_pos_class, anc_file)
            pos_file = os.path.join(self.root, random_pos_class, pos_file)
            neg_file = os.path.join(self.root, random_neg_class, neg_file)

            triplets.append((anc_file,
                             pos_file,
                             neg_file,
                             random_pos_class,
                             random_neg_class))
        return triplets

    def __len__(self):
        return len(self.training_triplets)

    def __getitem__(self, i):
        triplet = self.training_triplets[i]

        images = []
        for img_path in triplet[:3]:
            img = io.imread(img_path)
            if img_path in self.bbox:
                x = int(self.bbox[img_path]['x'])
                y = int(self.bbox[img_path]['y'])
                w = int(self.bbox[img_path]['w'])
                h = int(self.bbox[img_path]['h'])
            else:
                x, y = 0, 0
                w = img.shape[1]
                y = img.shape[0]
            img = img[y:y+h, x:x+w:, :]
            images.append(img)

        anc_img, pos_img, neg_img = images

        anc_img = self.transform(anc_img)
        pos_img = self.transform(pos_img)
        neg_img = self.transform(neg_img)

        pos_class = self.ids_to_labels[triplet[3]]
        neg_class = self.ids_to_labels[triplet[4]]

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_class': pos_class,
            'neg_class': neg_class
        }
        return sample


class ScoringDataset(Dataset):
    def __init__(self, db, transform, bbox):
        self.db = db
        self.transform = transform
        self.bbox = pd.read_csv(bbox)
        self.bbox.set_index('path', inplace=True)
        self.bbox = self.bbox.to_dict(orient='index')

    def __len__(self):
        return len(self.db)

    def __getitem__(self, i):
        path = self.db[i]
        image = io.imread(path)

        if path in self.bbox:
            x = int(self.bbox[path]['x'])
            y = int(self.bbox[path]['y'])
            w = int(self.bbox[path]['w'])
            h = int(self.bbox[path]['h'])
        else:
            x, y = 0, 0
            w = image.shape[1]
            y = image.shape[0]

        image = image[y:y+h, x:x+w:, :]
        image = self.transform(image)
        return image


data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: expand2square(img)),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


data_transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: expand2square(img)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
