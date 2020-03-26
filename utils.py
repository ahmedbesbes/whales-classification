import math
import os
import json
import simplejson
from datetime import datetime
from pprint import pprint
import pandas as pd
from PIL import Image


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def log_experience(args):
    arguments = vars(args)
    time_id = str(datetime.now())
    arguments['date'] = time_id
    arguments['leaderboard_score'] = None
    arguments['tag'] = args.tag

    print('logging these arguments for the experience ...')
    pprint(arguments)
    print('----')

    output_folder = f'./output/{time_id}_{args.tag}/'
    os.makedirs(output_folder)

    with open(os.path.join(args.logs_experiences, f'{time_id}.json'), 'w') as f:
        f.write(simplejson.dumps(simplejson.loads(
            json.dumps(arguments)), indent=4, sort_keys=True))
        # json.dump(arguments, f)

    with open(os.path.join(output_folder, f'{time_id}.json'), 'w') as f:
        f.write(simplejson.dumps(simplejson.loads(
            json.dumps(arguments)), indent=4, sort_keys=True))
        # json.dump(arguments, f)

    return time_id, output_folder


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    def scaler(x): return 1.

    # Lambda function to calculate the LR
    def lr_lambda(it): return min_lr + (max_lr -
                                        min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda
