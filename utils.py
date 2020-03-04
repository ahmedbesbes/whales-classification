import os
from datetime import datetime
from pprint import pprint
import pandas as pd
from PIL import Image


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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

    print('logging these arguments for the experience ...')
    pprint(arguments)
    print('----')

    if not os.path.exists(args.logs_experiences):
        logs = pd.DataFrame([arguments])
        logs.to_csv(args.logs_experiences, index=False)
    else:
        logs = pd.read_csv(args.logs_experiences)
        logs = logs.append([arguments])
        logs.to_csv(args.logs_experiences, index=False)

    return time_id
