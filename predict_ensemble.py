import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from collections import namedtuple

import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import img_to_tensor
from albumentations import Compose
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from tools.config import load_config
from dataset.openeds_dataset import OpenEDSDatasetTest


ModelConfig = namedtuple("ModelConfig", "config_path weights_path weight")
configs = [
    ModelConfig("configs/se50.json", "/home/almaz/Downloads/segmentation_scse_unet_seresnext50_0_best_miou.pt", 1),
]


def predict_ensemble(args):

    preds_dict = {}

    for model_config in configs:

        transforms = Compose([
        ])
        dataset = OpenEDSDatasetTest(data_path=args.data_path,
                                     labels_file=args.label_file,
                                     save_path=args.save_dir,
                                     transforms=transforms,
                                     normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                                     cumulative=True)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=False)

        conf = load_config(model_config.config_path)
        models_zoo = conf.get('models_zoo', 'selim')
        if models_zoo == 'qubvel':
            import segmentation_models_pytorch as smp
            model = smp.Unet(encoder_name=conf['encoder'], classes=conf['num_classes'])
        else:
            model = models.__dict__[conf['network']](seg_classes=4, backbone_arch=conf['encoder'])
        model = torch.nn.DataParallel(model).cuda()

        checkpoint_path = model_config.weights_path
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        with torch.no_grad():
            for sample in tqdm(data_loader):

                imgs = sample["image"].cuda().float()
                output = model(imgs)
                output_flip = torch.flip(model(torch.flip(imgs, dims=(3,))), dims=(3,))

                output = (output + output_flip) / 2
                output = output.cpu()

                for i in range(output.shape[0]):
                    img_name = sample["img_name"][i]

                    if img_name not in preds_dict:
                        preds_dict[img_name] = {'output': output[i] * model_config.weight,
                                                'total_weight': model_config.weight}
                    else:
                        preds_dict[img_name]['output'] += output[i] * model_config.weight
                        preds_dict[img_name]['total_weight'] += model_config.weight

    return preds_dict


def normalize_preds(preds_dict):

    for img_name, output_dict in preds_dict.items():
        average = output_dict['output'] / output_dict['total_weight']
        softmax = torch.softmax(average, dim=0)
        argmax = torch.argmax(softmax, dim=0)
        preds_dict[img_name] = argmax

    return preds_dict


def save_preds(args, preds_dict):

    for img_name, pred in preds_dict.items():

        img_save_path = os.path.join(args.save_dir,
                                     img_name.replace('label_', ''))
        np.save(img_save_path, pred.cpu().numpy().astype(np.uint8)[8:-8, ...])


def main():
    parser = argparse.ArgumentParser("OpenEDS One-by-One Predictor")
    arg = parser.add_argument
    arg('--data-path', type=str, default='/media/almaz/1tb/openeds/openEDS2020-SparseSegmentation/participant')
    arg('--label-file', type=str,
        default='fold_0_val.txt',
        help='Text file with all images for inference')
    arg('--save-dir', type=str, default='/media/almaz/1tb/openeds/predictions/ensemble')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    preds_total = predict_ensemble(args)
    preds_total = normalize_preds(preds_total)
    save_preds(args, preds_total)


if __name__ == '__main__':
    main()
