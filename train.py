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
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from sklearn.metrics.pairwise import cosine_similarity

from backbones.resnet_models import ResNetModels
from backbones.densenet_models import DenseNetModels
from backbones import model_factory
from dataloader import WhalesData, augmentation
from sampler import PKSampler, PKSampler2
from utils import get_lr, set_lr, log_experience, cyclical_lr
from losses import TripletLoss

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data', default='/data_science/computer_vision/whales/data/data_full_new.csv', type=str)
parser.add_argument(
    '--root', default='/data_science/computer_vision/whales/data/new_data/train/', type=str)
parser.add_argument(
    '--root-test', default='/data_science/computer_vision/whales/data/test_val/', type=str)
parser.add_argument('--bbox-train', type=str,
                    default='/data_science/computer_vision/whales/bounding_boxes/train_bbox.csv')
parser.add_argument('--bbox-test', type=str,
                    default='/data_science/computer_vision/whales/bounding_boxes/test_bbox.csv')
parser.add_argument('--bbox-all', type=str,
                    default='/data_science/computer_vision/whales/bounding_boxes/all_bbox.csv')
parser.add_argument('--pseudo-labels', type=str,
                    default="/data_science/computer_vision/whales/data/bootstrapped_data.csv")

parser.add_argument('--crop', type=int, default=1, choices=[0, 1])
parser.add_argument('--pseudo-label', type=int, choices=[0, 1], default=0)

parser.add_argument('--archi', default='resnet34',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnext',
                             'densenet121',
                             'mobilenet',
                             'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], type=str)
parser.add_argument('--embedding-dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--alpha', type=int, default=8)
parser.add_argument('--pretrained', type=int, choices=[0, 1], default=1)
parser.add_argument('--image-size', type=int, default=224)
parser.add_argument('--freeze', type=int, default=0, choices=[0, 1, 2])
parser.add_argument('--gap', type=int, choices=[0, 1], default=1)
parser.add_argument('--heavy', type=int, choices=[0, 1], default=1)

parser.add_argument('--margin', type=float, default=-1)
parser.add_argument('-p', type=int, default=16)
parser.add_argument('-k', type=int, default=4)
parser.add_argument('--sampler', type=int, default=2, choices=[1, 2])

parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--wd', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num-workers', type=int, default=12)

parser.add_argument('--logging-step', type=int, default=10)
parser.add_argument('--output', type=str, default='./models/')
parser.add_argument('--submissions', type=str, default='./submissions/')
parser.add_argument('--logs-experiences', type=str,
                    default='./experiences/')


parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--pop-fc', type=int, default=1)
parser.add_argument('--flush', type=int, choices=[0, 1], default=1)
parser.add_argument('--log_path', type=str, default='./logs/')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--save-optim', type=int, default=0, choices=[0, 1])
parser.add_argument('--load-optim', type=int, default=0, choices=[0, 1])

parser.add_argument('--checkpoint-period', type=int, default=-1)


parser.add_argument('--scheduler', type=str,
                    choices=['multistep', 'clr', 'warmup'])
parser.add_argument('--min-lr', type=float, default=2.4e-6)
parser.add_argument('--max-lr', type=float, default=1.4e-5)
parser.add_argument('--step-size', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', nargs='+', type=int)
parser.add_argument('--lr-end', type=float, default=None)
parser.add_argument('--warmup-epochs', type=int, default=2)


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

    data_transform = augmentation(args.image_size,
                                  train=True,
                                  heavy=bool(args.heavy))

    time_id, output_folder = log_experience(args, data_transform)

    data = pd.read_csv(args.data)
    if bool(args.pseudo_label):
        bootstrapped_data = pd.read_csv(args.pseudo_labels)
        data = pd.concat([data, bootstrapped_data], axis=0)
        data['file_id'] = data.index.tolist()

    mapping_filename_path = dict(zip(data['filename'], data['full_path']))
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
        'dropout': args.dropout,
        'alpha': args.alpha,
        'gap': args.gap
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

    if (args.freeze == 1) and (args.archi.startswith('resnet')):
        for param in model.model.layer1.parameters():
            param.requires_grad = False

    elif (args.freeze == 2) and (args.archi.startswith('resnet')):
        for param in model.model.layer1.parameters():
            param.requires_grad = False
        for param in model.model.layer2.parameters():
            param.requires_grad = False

    dataset = WhalesData(paths=paths,
                         bbox=args.bbox_train,
                         mapping_label_id=mapping_label_id,
                         transform=data_transform,
                         crop=bool(args.crop))

    if args.sampler == 1:
        sampler = PKSampler(root=args.root,
                            data_source=dataset,
                            classes=classes,
                            labels_to_samples=labels_to_samples,
                            mapping_files_to_global_id=mapping_files_to_global_id,
                            mapping_filename_path=mapping_filename_path,
                            p=args.p,
                            k=args.k)
    elif args.sampler == 2:
        sampler = PKSampler2(root=args.root,
                             data_source=dataset,
                             classes=classes,
                             labels_to_samples=labels_to_samples,
                             mapping_files_to_global_id=mapping_files_to_global_id,
                             mapping_filename_path=mapping_filename_path,
                             p=args.p,
                             k=args.k)

    dataloader = DataLoader(dataset,
                            batch_size=args.p*args.k,
                            sampler=sampler,
                            drop_last=True,
                            num_workers=args.num_workers)

   # define loss

    if args.margin == -1:
        criterion = TripletLoss(margin='soft', sample=False)
    else:
        criterion = TripletLoss(margin=args.margin, sample=False)

    # define scheduler

    if args.scheduler == 'clr':
        print('Using cyclic learning rate scheduler')
        print(
            f'Step size: {args.step_size} | min_lr: {args.min_lr} | max_lr: {args.max_lr}')
        optimizer = Adam(model.parameters(), lr=1, weight_decay=args.wd)
        step_size = int(len(dataloader) * args.step_size)
        clr = cyclical_lr(step_size,
                          args.min_lr,
                          args.max_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

    else:
        # define / load optimizer

        if (args.weights is not None) & (bool(args.load_optim)):
            optimizer = torch.load(args.weights)['optimizer']
        else:
            parameters = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = Adam(parameters,
                             lr=args.lr,
                             weight_decay=args.wd)

        # multi step

        if args.scheduler == 'multistep':
            print('Using multistep scheduler')
            print(f'gamma: {args.gamma} | milestone: {args.milestones}')
            scheduler = MultiStepLR(optimizer,
                                    milestones=args.milestones,
                                    gamma=args.gamma)

        # warmp + cosine annealing

        elif args.scheduler == 'warmup':
            print(f'Using warmup scheduler with cosine annealing')
            print(
                f'warmup epochs : {args.warmup_epochs} | total epochs {args.epochs}')
            print(f'lr_start : {args.lr} ---> lr_end : {args.lr_end}')

            scheduler_cosine = CosineAnnealingLR(optimizer,
                                                 args.epochs,
                                                 eta_min=args.lr_end)
            scheduler = GradualWarmupScheduler(optimizer,
                                               multiplier=1,
                                               total_epoch=args.warmup_epochs,
                                               after_scheduler=scheduler_cosine
                                               )
    model.train()

    for epoch in tqdm(range(args.start_epoch, args.epochs + args.start_epoch)):
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
            'scheduler': scheduler,
            'output_folder': output_folder
        }
        _ = train(**params)

        if args.scheduler in ['multistep', 'warmup']:
            scheduler.step()

    if bool(args.save_optim):
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
    else:
        state = {
            'state_dict': model.state_dict()
        }
    torch.save(state,
               os.path.join(output_folder,
                            f'{time_id}.pth'))

    compute_predictions(model, mapping_label_id, time_id, output_folder)


def train(model, dataloader, optimizer, criterion, logging_step, epoch, epochs, writer, time_id, output_folder, scheduler):
    current_lr = get_lr(optimizer)
    losses = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        images = batch['image']
        if np.random.rand() > 0.5:
            images = torch.flip(images, [-1])
        targets = batch['label']

        predictions = model(images.cuda())
        optimizer.zero_grad()
        loss = criterion(predictions, targets.cuda())

        loss.backward()

        if args.scheduler == 'clr':
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

        if bool(args.save_optim):
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

        else:
            state = {
                'state_dict': model.state_dict()
            }
        torch.save(state,
                   os.path.join(output_folder,
                                f'{time_id}_pth'))

    return average_loss


def compute_predictions(model, mapping_label_id, time_id, output_folder):
    model.eval()
    print("generating predictions ......")
    db = []
    train_folder = args.root
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
                                    num_workers=11,
                                    batch_size=args.p * args.k)

    embeddings = []
    for batch in tqdm(scoring_dataloader, total=len(scoring_dataloader)):
        with torch.no_grad():
            embedding = model(batch['image'].cuda())
            embedding = embedding.cpu().detach().numpy()
            embeddings.append(embedding)
    embeddings = np.concatenate(embeddings)

    np.save(os.path.join(output_folder,
                         f'embeddings_{time_id}.npy'),
            embeddings)

    test_dataset = WhalesData(test_db,
                              args.bbox_test,
                              mapping_label_id,
                              data_transform_test,
                              crop=bool(args.crop),
                              test=True)
    test_dataloader = DataLoader(test_dataset,
                                 num_workers=11,
                                 shuffle=False,
                                 batch_size=args.p * args.k)

    test_embeddings = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        with torch.no_grad():
            embedding = model(batch['image'].cuda())
            embedding = embedding.cpu().detach().numpy()
            test_embeddings.append(embedding)

    test_embeddings = np.concatenate(test_embeddings)

    np.save(os.path.join(output_folder,
                         f'embeddings_test_{time_id}.npy'),
            test_embeddings)

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
        output_folder, f'{time_id}.csv'), header=None, sep=',', index=False)
    print("predictions generated...")


if __name__ == "__main__":
    main()
