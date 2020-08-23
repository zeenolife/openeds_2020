import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from losses import miou_round


if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenEDS Validator")
    arg = parser.add_argument
    arg('--data-dir', type=str, default='/data/openeds/openEDS2020-SparseSegmentation/participant')
    arg('--pred-dir', type=str, default='/data/openeds/predictions/ensemble')
    arg('--fold', type=int, default='0')

    args = parser.parse_args()

    with open(os.path.join(args.data_path, 'fold_{}_val.txt'.format(args.fold))) as f:
        val_labels = [_.strip() for _ in f.readlines()]

    mious = []
    for val_label in tqdm(val_labels):

        true = np.load(os.path.join(args.data_path, val_label))
        pred = np.load(os.path.join(args.pred_path, val_label.replace('label_', '')))
        true = np.expand_dims(true, axis=0)
        pred = np.expand_dims(pred, axis=0)

        true, pred = torch.from_numpy(true).float(), torch.from_numpy(pred).float()

        miou = miou_round(true, pred).item()
        mious.append(miou)

    print('mIoU in fold {} - {:.6f}'.format(args.fold, np.mean(mious)))