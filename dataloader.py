import os
import numpy as np
from tqdm import tqdm
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WhalesDataset(Dataset):
    def __init__(self, num_triplets, transform, root, stats_classes, mapping_class_to_images):

        self.num_triplets = num_triplets
        self.classes = os.listdir('../data/train/')
        self.stats_classes = stats_classes
        self.mapping_class_to_images = mapping_class_to_images
        self.training_triplets = self.create_triplets()
        self.transform = transform

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

            anc_file = f'../data/train/{random_pos_class}/{anc_file}'
            pos_file = f'../data/train/{random_pos_class}/{pos_file}'
            neg_file = f'../data/train/{random_neg_class}/{neg_file}'

            triplets.append((anc_file, pos_file, neg_file))
        return triplets

    def __len__(self):
        return len(self.training_triplets)

    def __getitem__(self, i):
        triplet = self.training_triplets[i]

        anc_img = io.imread(triplet[0])
        pos_img = io.imread(triplet[1])
        neg_img = io.imread(triplet[2])

        anc_img = self.transform(anc_img)
        pos_img = self.transform(pos_img)
        neg_img = self.transform(neg_img)

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img
        }
        return sample


class ScoringDataset(Dataset):
    def __init__(self, db, transform):
        self.db = db
        self.transform = transform

    def __len__(self):
        return len(self.db)

    def __getitem__(self, i):
        path = self.db[i]
        image = io.imread(path)
        image = self.transform(image)
        return image


def data_transform(img):
    pad = abs((img.shape[0] - img.shape[1])) // 2
    min_length = np.argmin(img.shape[:2])
    if min_length == 0:
        pad = (0, pad)
    else:
        pad = (pad, 0)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(pad),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    return transform(img)


def data_transform_test(img):
    pad = abs((img.shape[0] - img.shape[1])) // 2
    min_length = np.argmin(img.shape[:2])
    if min_length == 0:
        pad = (0, pad)
    else:
        pad = (pad, 0)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(pad),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    return transform(img)
