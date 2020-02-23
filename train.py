import os
import shutil
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet34
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import faiss

from backbones.Resnet34 import FaceNetModel
from backbones.InceptionResnet import InceptionResnetV1
from dataloader import WhalesDataset, ScoringDataset, data_transform, data_transform_test
from utils import get_lr
from losses import TripletLoss

parser = argparse.ArgumentParser()
parser.add_argument(
    '--root', default='/data_science/computer_vision/whales/data/', type=str)

parser.add_argument('--archi', default='resnet34',
                    choices=['resnet34', 'inception'], type=str)
parser.add_argument('--embedding-dim', type=int, default=128)
parser.add_argument('--pretrained', type=int, choices=[0, 1], default=1)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--num-triplets', type=int, default=10000)

parser.add_argument('--learning-rate', type=float, default=3e-4)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num-workers', type=int, default=8)

parser.add_argument('--logging-step', type=int, default=25)
parser.add_argument('--output', type=str, default='./models/')
parser.add_argument('--submissions', type=str, default='./submissions/')

np.random.seed(0)
torch.manual_seed(0)

args = parser.parse_args()
l2_dist = PairwiseDistance(2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    mapping_class_id = {}

    for i, c in enumerate(os.listdir(os.path.join(args.root, 'train'))):
        mapping_class_id[c] = i

    mapping_image_id = {}
    mapping_class_to_images = {}
    for c in os.listdir(os.path.join(args.root, 'train')):
        mapping_class_to_images[c] = os.listdir(
            os.path.join(args.root, 'train', c))
        for img in os.listdir(
                os.path.join(args.root, 'train', c)):
            mapping_image_id[img] = c

    stats_classes = pd.DataFrame()
    stats_classes['class'] = os.listdir(os.path.join(args.root, 'train'))
    stats_classes['num'] = stats_classes['class'].map(lambda c: len(os.listdir(
        os.path.join(args.root, 'train', c))))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.archi == 'resnet34':
        model = FaceNetModel(args.embedding_dim,
                             num_classes=len(mapping_class_to_images),
                             pretrained=bool(args.pretrained))
    elif args.archi == 'inception':
        model = InceptionResnetV1(num_classes=len(mapping_class_to_images),
                                  embedding_dim=args.embedding_dim)

    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = MultiStepLR(optimizer,
                            milestones=[5, 10, 15, 20, 25, 30],
                            gamma=0.2)

    for epoch in tqdm(range(args.epochs)):
        scheduler.step()
        current_lr = get_lr(optimizer)

        dataset = WhalesDataset(args.num_triplets,
                                transform=data_transform,
                                root=args.root,
                                stats_classes=stats_classes,
                                mapping_class_to_images=mapping_class_to_images)
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

        params = {
            'model': model,
            'dataloader': dataloader,
            'optimizer': optimizer,
            'logging_step': args.logging_step,
            'epoch': epoch,
            'epochs': args.epochs,
            'current_lr': current_lr
        }
        train(**params)

    compute_predictions(model)


def train(model, dataloader, optimizer, logging_step, epoch, epochs, current_lr):
    losses = []
    for i, batch_sample in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        anc_img = batch_sample['anc_img'].to(device)
        pos_img = batch_sample['pos_img'].to(device)
        neg_img = batch_sample['neg_img'].to(device)

        with torch.set_grad_enabled(True):
            anc_embed, pos_embed, neg_embed = model(
                anc_img), model(pos_img), model(neg_img)

            # choose the hard negatives only for "training"
            pos_dist = l2_dist.forward(anc_embed, pos_embed)
            neg_dist = l2_dist.forward(anc_embed, neg_embed)

            all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
            hard_triplets = np.where(all == 1)
            if len(hard_triplets[0]) == 0:
                continue

            anc_hard_embed = anc_embed[hard_triplets].to(device)
            pos_hard_embed = pos_embed[hard_triplets].to(device)
            neg_hard_embed = neg_embed[hard_triplets].to(device)

            triplet_loss = TripletLoss(args.margin).forward(
                anc_hard_embed, pos_hard_embed, neg_hard_embed).to(device)

            optimizer.zero_grad()
            triplet_loss.backward()
            optimizer.step()

            losses.append(triplet_loss.item())

            if i % logging_step == 0:
                avg_running_loss = np.mean(losses)
                print(
                    f'[{epoch + 1} / {epochs}][{i} / {len(dataloader)}][lr: {current_lr}] loss = {avg_running_loss}')
    avg_loss = np.mean(losses)
    torch.save({'loss': avg_loss, 'state_dict': model.state_dict()},
               os.path.join(args.output, 'triplet_loss_baseline.pth'))


def compute_predictions(model):
    print("generating predictions ...")
    db = []
    train_folder = os.path.join(args.root, 'train')
    for c in os.listdir(train_folder):
        for f in os.listdir(os.path.join(train_folder, c)):
            db.append(os.path.join(train_folder, c, f))

    db += [os.path.join(args.root, 'test_val', f)
           for f in os.listdir(os.path.join(args.root, 'test_val'))]
    test_db = sorted(
        [os.path.join(args.root, 'test_val', f) for f in os.listdir(os.path.join(args.root, 'test_val'))])

    scoring_dataset = ScoringDataset(db, data_transform_test)
    scoring_dataloader = DataLoader(
        scoring_dataset, shuffle=False, num_workers=10)

    embeddings = []
    for images in tqdm(scoring_dataloader, total=len(scoring_dataloader)):
        with torch.no_grad():
            embedding = model(images.cuda())
            embedding = embedding.cpu().detach().numpy()
            embeddings.append(embedding)
    embeddings = np.concatenate(embeddings)

    test_dataset = ScoringDataset(test_db, data_transform_test)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    test_embeddings = []
    for images in tqdm(test_dataloader, total=len(test_dataloader)):
        with torch.no_grad():
            embedding = model(images.cuda())
            embedding = embedding.cpu().detach().numpy()
            test_embeddings.append(embedding)

    test_embeddings = np.concatenate(test_embeddings)

    quantizer = faiss.IndexFlatL2(args.embedding_dim)
    faiss_index = faiss.IndexIVFFlat(
        quantizer, args.embedding_dim, 50, faiss.METRIC_INNER_PRODUCT)
    faiss_index.train(embeddings)
    faiss_index.add(embeddings)
    D, I = faiss_index.search(test_embeddings, 21)
    submission = pd.DataFrame(I)
    for c in submission.columns:
        submission[c] = submission[c].map(lambda v: db[v].split('/')[-1])

    submission.to_csv(os.path.join(
        args.submissions, 'triplet_loss_baseline.csv'), header=None, sep=',', index=False)
    print("predictions generated...")


if __name__ == "__main__":
    main()
