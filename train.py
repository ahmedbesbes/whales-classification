import os
import sys
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
from utils import get_lr, log_experience
from losses import TripletLoss

parser = argparse.ArgumentParser()
parser.add_argument(
    '--root', default='/data_science/computer_vision/whales/data/train/', type=str)
parser.add_argument(
    '--root-test', default='/data_science/computer_vision/whales/data/test_val/', type=str)

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
parser.add_argument('--step-size', type=float, default=5)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--hard', type=int, choices=[0, 1], default=1)
parser.add_argument('--classif', type=str,
                    choices=['binary', 'multi'], default='binary')
parser.add_argument('--weight-triplet', type=float, default=1)
parser.add_argument('--use-ce', type=int, default=1, choices=[0, 1])

parser.add_argument('--logging-step', type=int, default=25)
parser.add_argument('--output', type=str, default='./models/')
parser.add_argument('--submissions', type=str, default='./submissions/')
parser.add_argument('--logs-experiences', type=str,
                    default='./experiences/logs.csv')

parser.add_argument('--bbox-train', type=str,
                    default='/data_science/computer_vision/whales/bounding_boxes/train_bbox.csv')
parser.add_argument('--bbox-test', type=str,
                    default='/data_science/computer_vision/whales/bounding_boxes/test_bbox.csv')
parser.add_argument('--bbox-all', type=str,
                    default='/data_science/computer_vision/whales/bounding_boxes/all_bbox.csv')

parser.add_argument('--checkpoint', type=str, default=None)

np.random.seed(0)
torch.manual_seed(0)

args = parser.parse_args()
l2_dist = PairwiseDistance(2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    log_experience(args)

    mapping_class_id = {}

    for i, c in enumerate(os.listdir(args.root)):
        mapping_class_id[c] = i

    mapping_image_id = {}
    mapping_class_to_images = {}
    for c in os.listdir(args.root):
        mapping_class_to_images[c] = os.listdir(
            os.path.join(args.root, c))
        for img in os.listdir(
                os.path.join(args.root, c)):
            mapping_image_id[img] = c

    stats_classes = pd.DataFrame()
    stats_classes['class'] = os.listdir(args.root)
    stats_classes['num'] = stats_classes['class'].map(lambda c: len(os.listdir(
        os.path.join(args.root, c))))
    stats_classes['p'] = stats_classes['num'] / stats_classes['num'].sum()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.classif == "binary":
        num_classes = 2
        ids_to_labels = {}
        for c in mapping_class_id.keys():
            if c != '-1':
                ids_to_labels[c] = 0
            else:
                ids_to_labels[c] = 1

    elif args.classif == "multi":
        num_classes = len(mapping_class_id)
        ids_to_labels = mapping_class_id

    if args.checkpoint is not None:
        model = FaceNetModel(args.embedding_dim,
                             num_classes=num_classes,
                             pretrained=bool(args.pretrained))
        weights = torch.load(args.checkpoint)
        model.load_state_dict(weights.state_dict())
        print('loading saved model ...')
    else:
        if args.archi == 'resnet34':
            model = FaceNetModel(args.embedding_dim,
                                 num_classes=num_classes,
                                 pretrained=bool(args.pretrained))
        elif args.archi == 'inception':
            model = InceptionResnetV1(num_classes=num_classes,
                                      embedding_dim=args.embedding_dim)

    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in tqdm(range(args.epochs)):
        scheduler.step()
        current_lr = get_lr(optimizer)

        dataset = WhalesDataset(args.num_triplets,
                                transform=data_transform,
                                root=args.root,
                                stats_classes=stats_classes,
                                mapping_class_to_images=mapping_class_to_images,
                                ids_to_labels=ids_to_labels,
                                bbox=args.bbox_train)
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
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
    t_losses = []
    ce_losses = []

    for i, batch_sample in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        anc_img = batch_sample['anc_img'].to(device)
        pos_img = batch_sample['pos_img'].to(device)
        neg_img = batch_sample['neg_img'].to(device)

        with torch.set_grad_enabled(True):

            if bool(args.use_ce):
                anc_pred = model.forward_classifier(anc_img)
                pos_pred = model.forward_classifier(pos_img)
                neg_pred = model.forward_classifier(neg_img)

                pos_targets = batch_sample['pos_class'].to(device)
                neg_targets = batch_sample['neg_class'].to(device)

                preds = torch.cat([anc_pred, pos_pred, neg_pred], dim=0)
                targets = torch.cat([pos_targets, pos_targets, neg_targets])

                ce_loss = nn.CrossEntropyLoss()(preds, targets)

            anc_embed, pos_embed, neg_embed = model(
                anc_img), model(pos_img), model(neg_img)

            # choose the hard negatives only for "training"
            pos_dist = l2_dist.forward(anc_embed, pos_embed)
            neg_dist = l2_dist.forward(anc_embed, neg_embed)

            if bool(args.hard):
                all = (neg_dist - pos_dist <
                       args.margin).cpu().numpy().flatten()
                hard_triplets = np.where(all == 1)
                if len(hard_triplets[0]) == 0:
                    continue

                anc_hard_embed = anc_embed[hard_triplets].to(device)
                pos_hard_embed = pos_embed[hard_triplets].to(device)
                neg_hard_embed = neg_embed[hard_triplets].to(device)

                triplet_loss = TripletLoss(args.margin).forward(anc_hard_embed,
                                                                pos_hard_embed,
                                                                neg_hard_embed).to(device)
            else:
                triplet_loss = TripletLoss(args.margin).forward(anc_embed,
                                                                pos_embed,
                                                                neg_embed).to(device)
            if bool(args.use_ce):
                total_loss = ce_loss + args.weight_triplet * triplet_loss
            else:
                total_loss = triplet_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if bool(args.use_ce):
                ce_losses.append(ce_loss.item())
            t_losses.append(triplet_loss.item())
            losses.append(total_loss.item())

            if i % logging_step == 0:
                avg_running_loss = np.mean(losses)
                avg_running_t_loss = np.mean(t_losses)

                if bool(args.use_ce):
                    avg_running_ce_loss = np.mean(ce_losses)
                    print(f'[{epoch + 1} / {epochs}][{i} / {len(dataloader)}][lr: {current_lr}] losses: ce = {avg_running_ce_loss}| triplet = {avg_running_t_loss} | total: {avg_running_loss}')
                else:
                    print(
                        f'[{epoch + 1} / {epochs}][{i} / {len(dataloader)}][lr: {current_lr}] total: {avg_running_loss}')

    avg_loss = np.mean(losses)
    torch.save({'loss': avg_loss, 'state_dict': model.state_dict()},
               os.path.join(args.output, 'triplet_loss_baseline.pth'))


def compute_predictions(model):
    print("generating predictions ...")
    db = []
    train_folder = os.path.join(args.root)
    for c in os.listdir(train_folder):
        for f in os.listdir(os.path.join(train_folder, c)):
            db.append(os.path.join(train_folder, c, f))

    db += [os.path.join(args.root_test, f) for f in os.listdir(args.root_test)]
    test_db = sorted(
        [os.path.join(args.root_test, f) for f in os.listdir(args.root_test)])

    scoring_dataset = ScoringDataset(db, data_transform_test, args.bbox_all)
    scoring_dataloader = DataLoader(
        scoring_dataset, shuffle=False, num_workers=10)

    embeddings = []
    for images in tqdm(scoring_dataloader, total=len(scoring_dataloader)):
        with torch.no_grad():
            embedding = model(images.cuda())
            embedding = embedding.cpu().detach().numpy()
            embeddings.append(embedding)
    embeddings = np.concatenate(embeddings)

    test_dataset = ScoringDataset(test_db, data_transform_test, args.bbox_test)
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
