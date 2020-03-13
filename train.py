import os
import sys
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
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics.pairwise import cosine_similarity

from backbones.resnet_models import ResNetModels
from backbones.densenet_models import DenseNetModels
from backbones import model_factory
from dataloader import WhalesData, augmentation
from sampler import PKSampler, PKSampler2
from utils import get_lr, log_experience, cyclical_lr
from losses import TripletLoss

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data', default='/data_science/computer_vision/whales/data/data_full.csv', type=str)
parser.add_argument(
    '--root', default='/data_science/computer_vision/whales/data/train/', type=str)
parser.add_argument(
    '--root-test', default='/data_science/computer_vision/whales/data/test_val/', type=str)
parser.add_argument('--crop', type=int, default=1, choices=[0, 1])

parser.add_argument('--archi', default='resnet34',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121', 'mobilenet'], type=str)
parser.add_argument('--embedding-dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--pretrained', type=int, choices=[0, 1], default=1)
parser.add_argument('--image-size', type=int, default=224)

parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('-p', type=int, default=8)
parser.add_argument('-k', type=int, default=4)
parser.add_argument('--sampler', type=int, default=1, choices=[1, 2])

parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num-workers', type=int, default=11)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', nargs='+', type=int)

parser.add_argument('--logging-step', type=int, default=10)
parser.add_argument('--output', type=str, default='./models/')
parser.add_argument('--submissions', type=str, default='./submissions/')
parser.add_argument('--logs-experiences', type=str,
                    default='./experiences/')

parser.add_argument('--bbox-train', type=str,
                    default='/data_science/computer_vision/whales/bounding_boxes/train_bbox.csv')
parser.add_argument('--bbox-test', type=str,
                    default='/data_science/computer_vision/whales/bounding_boxes/test_bbox.csv')
parser.add_argument('--bbox-all', type=str,
                    default='/data_science/computer_vision/whales/bounding_boxes/all_bbox.csv')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--flush', type=int, choices=[0, 1], default=1)
parser.add_argument('--log_path', type=str, default='./logs/')

parser.add_argument('--checkpoint-period', type=int, default=-1)

parser.add_argument('--clr', action='store_true')
parser.add_argument('--min-lr', type=float, default=2.4e-6)
parser.add_argument('--max-lr', type=float, default=1.4e-5)
parser.add_argument('--step-size', type=int, default=4)

np.random.seed(0)
torch.manual_seed(0)

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    if args.flush == 1:
        objects = os.listdir(args.log_path)
        for f in objects:
            if os.path.isdir(os.path.join(args.log_path, f)):
                shutil.rmtree(os.path.join(args.log_path, f))
    now = datetime.now()
    date = now.strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(args.log_path, date)
    os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    time_id = log_experience(args)

    data = pd.read_csv(args.data)
    classes = data.folder.unique()
    mapping_label_id = dict(zip(classes, range(len(classes))))

    num_classes = data.folder.nunique()
    mapping_files_to_global_id = dict(
        zip(data.full_path.tolist(), data.file_id.tolist()))
    paths = data.full_path.tolist()
    labels_to_samples = data.groupby('folder').agg(list)['filename'].to_dict()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_params = {
        'embedding_dim': args.embedding_dim,
        'num_classes': num_classes,
        'image_size': args.image_size,
        'archi': args.archi,
        'pretrained': bool(args.pretrained),
        'dropout': args.dropout
    }

    if args.checkpoint is not None:
        model = model_factory.get_model(**model_params)
        weights = torch.load(args.checkpoint)
        model.load_state_dict(weights)
        print('loading saved model ...')
    else:
        model = model_factory.get_model(**model_params)
        if args.weights is not None:
            print('loading pre-trained weights and changing input size ...')
            weights = torch.load(args.weights)
            weights.pop('model.fc.weight')
            weights.pop('model.fc.bias')
            weights.pop('model.classifier.weight')
            weights.pop('model.classifier.bias')
            model.load_state_dict(weights, strict=False)

    model.to(device)

    data_transform = augmentation(args.image_size, train=True)
    dataset = WhalesData(paths=paths,
                         bbox=args.bbox_train,
                         mapping_label_id=mapping_label_id,
                         transform=data_transform,
                         crop=bool(args.crop)
                         )

    if args.sampler == 1:
        sampler = PKSampler(root=args.root,
                            data_source=dataset,
                            classes=classes,
                            labels_to_samples=labels_to_samples,
                            mapping_files_to_global_id=mapping_files_to_global_id,
                            p=args.p,
                            k=args.k)
    elif args.sampler == 2:
        sampler = PKSampler2(root=args.root,
                             data_source=dataset,
                             classes=classes,
                             labels_to_samples=labels_to_samples,
                             mapping_files_to_global_id=mapping_files_to_global_id,
                             p=args.p,
                             k=args.k)

    dataloader = DataLoader(dataset,
                            batch_size=args.p*args.k,
                            sampler=sampler,
                            num_workers=args.num_workers)

    if args.margin == -1:
        criterion = TripletLoss(margin='soft', sample=False)
    else:
        criterion = TripletLoss(margin=args.margin, sample=False)

    if args.clr:
        print('using learning rate scheduling ...')
        optimizer = Adam(model.parameters(), lr=1, weight_decay=args.wd)
        step_size = int(len(dataloader) * args.step_size)
        clr = cyclical_lr(step_size,
                          args.min_lr,
                          args.max_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    else:
        optimizer = Adam(model.parameters(), lr=args.lr,
                         weight_decay=args.wd)
        scheduler = MultiStepLR(optimizer,
                                milestones=args.milestones,
                                gamma=args.gamma)

    model.train()

    for epoch in tqdm(range(args.epochs)):
        params = {
            'model': model,
            'dataloader': dataloader,
            'optimizer': optimizer,
            'criterion': criterion,
            'logging_step': args.logging_step,
            'epoch': epoch,
            'epochs': args.epochs,
            'writer': writer,
            'time_id': time_id,
            'scheduler': scheduler
        }
        _ = train(**params)

        if not args.clr:
            scheduler.step()

    torch.save(model.state_dict(),
               os.path.join(args.output,
                            f'{time_id}_pth'))

    compute_predictions(model, mapping_label_id, time_id)


def train(model, dataloader, optimizer, criterion, logging_step, epoch, epochs, writer, time_id, scheduler):
    current_lr = get_lr(optimizer)
    losses = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        images = batch['image']
        targets = batch['label']
        predictions = model(images.cuda())
        optimizer.zero_grad()
        loss = criterion(predictions, targets.cuda())

        loss.backward()

        if args.clr:
            scheduler.step()

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
        torch.save(model.state_dict(),
                   os.path.join(args.output,
                                f'{time_id}_pth'))

    return average_loss


def compute_predictions(model, mapping_label_id, time_id):
    model.eval()
    print("generating predictions ...")
    db = []
    train_folder = os.path.join(args.root)
    for c in os.listdir(train_folder):
        for f in os.listdir(os.path.join(train_folder, c)):
            db.append(os.path.join(train_folder, c, f))

    db += [os.path.join(args.root_test, f) for f in os.listdir(args.root_test)]
    test_db = sorted(
        [os.path.join(args.root_test, f) for f in os.listdir(args.root_test)])

    data_transform_test = augmentation(args.image_size, train=False)
    scoring_dataset = WhalesData(db,
                                 args.bbox_all,
                                 mapping_label_id,
                                 data_transform_test,
                                 crop=bool(args.crop),
                                 test=True)

    scoring_dataloader = DataLoader(scoring_dataset,
                                    shuffle=False,
                                    num_workers=10,
                                    batch_size=32)

    embeddings = []
    for batch in tqdm(scoring_dataloader, total=len(scoring_dataloader)):
        with torch.no_grad():
            embedding = model(batch['image'].cuda())
            embedding = embedding.cpu().detach().numpy()
            embeddings.append(embedding)
    embeddings = np.concatenate(embeddings)

    test_dataset = WhalesData(test_db,
                              args.bbox_test,
                              mapping_label_id,
                              data_transform_test,
                              crop=bool(args.crop),
                              test=True)
    test_dataloader = DataLoader(test_dataset,
                                 shuffle=False,
                                 batch_size=32)

    test_embeddings = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        with torch.no_grad():
            embedding = model(batch['image'].cuda())
            embedding = embedding.cpu().detach().numpy()
            test_embeddings.append(embedding)

    test_embeddings = np.concatenate(test_embeddings)

    csm = cosine_similarity(test_embeddings, embeddings)
    all_indices = []
    for i in range(len(csm)):
        test_file = test_db[i]
        index_test_file_all = db.index(test_file)
        similarities = csm[i]

        index_sorted_sim = np.argsort(similarities)[::-1]

        c = 0
        indices = []
        for idx in index_sorted_sim:
            if idx != index_test_file_all:
                indices.append(idx)
                c += 1
            if c > 20:
                break
        all_indices.append(indices)

    submission = pd.DataFrame(all_indices)
    submission = submission.rename(columns=dict(
        zip(submission.columns.tolist(), [c+1 for c in submission.columns.tolist()])))

    for c in submission.columns:
        submission[c] = submission[c].map(lambda v: db[v].split('/')[-1])

    submission[0] = [f.split('/')[-1] for f in test_db]
    submission = submission[range(21)]

    submission.to_csv(os.path.join(
        args.submissions, f'{time_id}.csv'), header=None, sep=',', index=False)
    print("predictions generated...")


if __name__ == "__main__":
    main()
