import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import expand2square


class WhalesDataset(Dataset):
    def __init__(self, num_triplets, transform, root, stats_classes, mapping_class_to_images):

        self.num_triplets = num_triplets
        self.root = root
        self.classes = os.listdir(self.root)
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

            anc_file = os.path.join(self.root, random_pos_class, anc_file)
            pos_file = os.path.join(self.root, random_pos_class, pos_file)
            neg_file = os.path.join(self.root, random_neg_class, neg_file)

            triplets.append((anc_file, pos_file, neg_file))
        return triplets

    def __len__(self):
        return len(self.training_triplets)

    def __getitem__(self, i):
        triplet = self.training_triplets[i]

        anc_img = Image.open(triplet[0])
        pos_img = Image.open(triplet[1])
        neg_img = Image.open(triplet[2])

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
        image = Image.open(path)
        image = self.transform(image)
        return image


data_transform = transforms.Compose([
    transforms.Lambda(lambda img: expand2square(img)),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


data_transform_test = transforms.Compose([
    transforms.Lambda(lambda img: expand2square(img)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
