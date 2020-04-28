import os
import sys
import shutil
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


from backbones.resnet_models import ResNetModels
from backbones.densenet_models import DenseNetModels
from backbones import model_factory
from dataloader import WhalesData, augmentation
from samplers import pk_sampler, pk_sample_full_coverage_epoch
from utils import (get_lr, set_lr, log_experience, parse_config,
                   get_scheduler, get_sampler, parse_arguments,
                   compute_predictions, get_summary_writer)
from losses.triplet_loss import TripletLoss
from losses.arcface import ArcMarginProduct


args = parse_arguments()
data_files = parse_config()

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    writer = get_summary_writer(args)
    data_transform = augmentation(args.image_size, train=True)

    time_id, output_folder = log_experience(args, data_transform)

    data = pd.read_csv(data_files['data'])
    if bool(args.use_pseudo_labels):
        pseudo_labels = pd.read_csv(data_files['pseudo_labels'])
        pseudo_labels = pseudo_labels[pseudo_labels['proba'] > 0.9]
        mapping_pseudo_files_folders = dict(
            zip(pseudo_labels['full_path'], pseudo_labels['folder']))
        print(f'using {len(pseudo_labels)} pseudo labels !')
        data = pd.concat(
            [data, pseudo_labels[['folder', 'full_path']]], axis=0, sort=False)
    else:
        mapping_pseudo_files_folders = None

    data.reset_index(inplace=True, drop=True)
    data['file_id'] = data.index.tolist()

    classes = data.folder.unique()
    mapping_label_id = dict(zip(classes, range(len(classes))))
    num_classes = data.folder.nunique()
    paths = data.full_path.tolist()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_params = {
        'embedding_dim': args.embedding_dim,
        'num_classes': num_classes,
        'image_size': args.image_size,
        'archi': args.archi,
        'pretrained': bool(args.pretrained),
        'dropout': args.dropout,
        'alpha': args.alpha
    }

    model = model_factory.get_model(**model_params)

    if args.weights is not None:
        print('loading pre-trained weights and changing input size ...')
        weights = torch.load(args.weights)
        if 'state_dict' in weights.keys():
            weights = weights['state_dict']
        if args.archi != 'densenet121':

            if bool(args.pop_fc):
                weights.pop('model.fc.weight')
                weights.pop('model.fc.bias')

            try:
                weights.pop('model.classifier.weight')
                weights.pop('model.classifier.bias')
            except:
                print('no classifier. skipping.')
        model.load_state_dict(weights, strict=False)
    model.to(device)

    dataset = WhalesData(paths=paths,
                         bbox=data_files['bbox_train'],
                         mapping_label_id=mapping_label_id,
                         mapping_pseudo_files_folders=mapping_pseudo_files_folders,
                         transform=data_transform,
                         crop=bool(args.crop))

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=args.num_workers)

   # define loss

    arcface_loss = ArcMarginProduct(args.embedding_dim, num_classes)
    arcface_loss.to(device)

    # define optimizer

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam([{'params': parameters},
                      {'params': arcface_loss.parameters()}],
                     lr=args.lr,
                     weight_decay=args.wd)

    # define scheduler

    scheduler = get_scheduler(args, optimizer)
    model.train()

    # start training loop

    for epoch in tqdm(range(args.start_epoch, args.epochs + args.start_epoch)):
        params = {
            'model': model,
            'dataloader': dataloader,
            'optimizer': optimizer,
            'criterion': arcface_loss,
            'logging_step': args.logging_step,
            'epoch': epoch,
            'epochs': args.epochs,
            'writer': writer,
            'time_id': time_id,
            'scheduler': scheduler,
            'output_folder': output_folder
        }
        _ = train(**params)

        scheduler.step()

    state = {
        'state_dict': model.state_dict()
    }
    torch.save(state,
               os.path.join(output_folder,
                            f'{time_id}.pth'))

    compute_predictions(args,
                        data_files,
                        model,
                        mapping_label_id,
                        time_id,
                        output_folder)


def train(model, dataloader, optimizer, criterion, logging_step, epoch, epochs, writer, time_id, output_folder, scheduler):
    current_lr = get_lr(optimizer)
    losses = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        images = batch['image'].cuda()
        targets = batch['label'].cuda()

        features, logits = model.forward_classifier(images)

        output = criterion(features, targets)
        loss_arcface = nn.CrossEntropyLoss()(output, targets)

        loss_ce = nn.CrossEntropyLoss()(logits, targets)
        loss = loss_arcface + loss_ce

        optimizer.zero_grad()
        loss.backward()

        current_lr = get_lr(optimizer)

        optimizer.step()
        losses.append(loss.item())

        writer.add_scalar(f'loss',
                          loss.item(),
                          epoch * len(dataloader) + i
                          )

        if (i % logging_step == 0) & (i > 0):
            running_avg_loss = np.mean(losses)
            print(
                f'[Epoch {epoch+1}][Batch {i} / {len(dataloader)}][lr: {current_lr}]: loss {running_avg_loss}')

    average_loss = np.mean(losses)
    writer.add_scalar(f'loss-epoch',
                      average_loss,
                      epoch
                      )

    if (args.checkpoint_period != -1) & (args.checkpoint_period % (epoch+1) == 0):
        state = {
            'state_dict': model.state_dict()
        }
        torch.save(state,
                   os.path.join(output_folder,
                                f'{time_id}_pth'))

    return average_loss


if __name__ == "__main__":
    main()
