import argparse
from configparser import ConfigParser
import math
import os
import shutil
import json
import simplejson
from datetime import datetime
from pprint import pprint

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import WhalesData, augmentation
from warmup_scheduler import GradualWarmupScheduler
from samplers import pk_sampler, pk_sample_full_coverage_epoch


# utility function to parse the arguments from the command line

def parse_arguments():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--gap', type=int, choices=[0, 1], default=1)

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
                        choices=['multistep', 'cosine', 'warmup'], default='warmup')
    parser.add_argument('--min-lr', type=float, default=2.4e-6)
    parser.add_argument('--max-lr', type=float, default=1.4e-5)
    parser.add_argument('--step-size', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--milestones', nargs='+', type=int)
    parser.add_argument('--lr-end', type=float, default=1e-6)
    parser.add_argument('--warmup-epochs', type=int, default=2)

    args = parser.parse_args()
    return args


# function to get a tensorboard summary writer

def get_summary_writer(args):
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
    return writer


# data parsing function

def parse_config():
    config_parser = ConfigParser()
    config_parser.read('./config.ini')
    section = config_parser['csv']
    items = list(section.items())
    items = dict(items)
    return items

# function to log parameters


def log_experience(args, data_transform):

    data_transform_repr = data_transform.indented_repr()
    arguments = vars(args)
    time_id = str(datetime.now())
    arguments['date'] = time_id
    arguments['leaderboard_score'] = None
    arguments['tag'] = args.tag
    arguments['_augmentation'] = data_transform_repr

    print('logging these arguments for the experience ...')
    pprint(arguments)
    print('----')

    output_folder = f'./output/{time_id}_{args.tag}/'
    os.makedirs(output_folder)

    with open(os.path.join(output_folder, f'{time_id}.json'), 'w') as f:
        f.write(simplejson.dumps(simplejson.loads(
            json.dumps(arguments)), indent=4, sort_keys=True))

    return time_id, output_folder


# utility function to convert an image to a square while keeping its aspect ratio

def expand2square(pil_img):
    background_color = (0, 0, 0)
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# utlity function to get and set learning rates in optimizer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# utility function to define learning rate schedulers

def get_scheduler(args, optimizer):
    args = vars(args)

    if args['scheduler'] == "warmup":
        print(f'Using warmup scheduler with cosine annealing')
        print(
            f"warmup epochs : {args['warmup_epochs']} | total epochs {args['epochs']}")
        print(f"lr_start : {args['lr']} ---> lr_end : {args['lr_end']}")

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                      args['epochs'],
                                                                      eta_min=args['lr_end'])
        scheduler = GradualWarmupScheduler(optimizer,
                                           multiplier=1,
                                           total_epoch=args['warmup_epochs'],
                                           after_scheduler=scheduler_cosine)

    elif args['scheduler'] == "multistep":
        print(
            f"Using multistep scheduler with gamma = {args['gamma']} and milestones = {args['milestones']}")

        scheduler = MultiStepLR(optimizer,
                                milestones=args['milestones'],
                                gamma=args['gamma'])

    elif args['scheduler'] == "cosine":
        print(f"Using cosine annealing from {args['lr']} to {args['lr_end']}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               args['epochs'],
                                                               eta_min=args['lr_end'])

    return scheduler

# utility function to define Dataset Samplers


def get_sampler(args,
                data_files,
                dataset,
                classes,
                labels_to_samples,
                mapping_files_to_global_id,
                mapping_filename_path):

    args = vars(args)
    if args['sampler'] == 1:
        sampler = pk_sampler.PKSampler(root=data_files["root"],
                                       data_source=dataset,
                                       classes=classes,
                                       labels_to_samples=labels_to_samples,
                                       mapping_files_to_global_id=mapping_files_to_global_id,
                                       mapping_filename_path=mapping_filename_path,
                                       p=args['p'],
                                       k=args['k'])

    elif args['sampler'] == 2:
        sampler = pk_sample_full_coverage_epoch.PKSampler(root=data_files['root'],
                                                          data_source=dataset,
                                                          classes=classes,
                                                          labels_to_samples=labels_to_samples,
                                                          mapping_files_to_global_id=mapping_files_to_global_id,
                                                          mapping_filename_path=mapping_filename_path,
                                                          p=args['p'],
                                                          k=args['k'])

    return sampler


# utility function to generate a submission file


def compute_predictions(args, data_files, model, mapping_label_id, time_id, output_folder):
    model.eval()
    print("generating predictions ......")
    db = []
    train_folder = data_files['root']
    for c in os.listdir(train_folder):
        for f in os.listdir(os.path.join(train_folder, c)):
            db.append(os.path.join(train_folder, c, f))

    db += [os.path.join(data_files['root_test'], f)
           for f in os.listdir(data_files['root_test'])]
    test_db = sorted(
        [os.path.join(data_files['root_test'], f) for f in os.listdir(data_files['root_test'])])

    data_transform_test = augmentation(args.image_size, train=False)
    scoring_dataset = WhalesData(db,
                                 data_files['bbox_all'],
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
                              data_files['bbox_test'],
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
