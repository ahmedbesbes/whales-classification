import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def detect(source, save_img=False):
    device = ""
    cfg = "./cfg/yolov3-1cls.cfg"
    names = "data/whales.names"
    weights = "./weights/last.pt"
    half = True
    img_size = 416
    conf_thres = 0.3
    iou_thres = 0.6
    classes = None
    agnostic_nms = False
    save_txt = False

    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    img_size = (320, 192) if ONNX_EXPORT else img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(
        device='cpu' if ONNX_EXPORT else device)

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()
    # torch_utils.model_info(model, report='summary')  # 'full' or 'summary'

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    save_img = True
    dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    outputs = []
    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0].float() if half else model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                predictions = det.cpu().detach().numpy()[0]
                output = {}
                output['prediction'] = predictions
                output['path'] = path
                outputs.append(output)
    return outputs
