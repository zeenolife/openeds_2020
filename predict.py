import argparse
import os
import warnings
import numpy as np
import torch
from albumentations import Compose
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from dataset.openeds_dataset import OpenEDSDatasetTest
from tools.config import load_config

warnings.simplefilter("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenEDS Predictor")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', default='configs/se50.json', help='path to configuration file')
    arg('--data-path', type=str, default='/media/almaz/1tb/openeds/openEDS2020-SparseSegmentation/participant')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--save-dir', type=str, default='/media/almaz/1tb/openeds/predictions/')
    arg('--save-name', type=str, default='se50/')
    arg('--model', type=str, default='')

    args = parser.parse_args()
    os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok=True)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    conf = load_config(args.config)
    model = models.__dict__[conf['network']](seg_classes=4, backbone_arch=conf['encoder'])
    model = torch.nn.DataParallel(model).cuda()
    print("=> loading checkpoint '{}'".format(args.model))
    checkpoint = torch.load(args.model, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    transforms = Compose([
    ])
    dataset = OpenEDSDatasetTest(data_path=args.data_path,
                                 labels_file='test.txt',
                                 save_path=os.path.join(args.save_dir, args.save_name),
                                 transforms=transforms,
                                 normalize=conf['input'].get('normalize', None))
    data_loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=False)
    with torch.no_grad():
        for sample in tqdm(data_loader):

            imgs = sample["image"].cuda().float()
            output = model(imgs)
            pred = torch.softmax(output, dim=1)
            argmax = torch.argmax(pred, dim=1)

            for i in range(output.shape[0]):
                img_save_path = os.path.join(args.save_dir, args.save_name, sample["img_name"][i])
                np.save(img_save_path, argmax[i].cpu().numpy().astype(np.uint8)[8:-8, ...])
