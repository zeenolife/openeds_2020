import os
import numpy as np


def main(labels_dir):

    with open(os.path.join(labels_dir, 'labels.txt')) as f:
        labels = [_.strip() for _ in f.readlines()]
        labels = set(labels)

    with open(os.path.join(labels_dir, 'images.txt')) as f:
        images = [_.strip() for _ in f.readlines()]
        images = [_.replace('.png', '.npy') for _ in images]
        images = [_.replace('/', '/label_') for _ in images]
        images = [_ for _ in images if _ not in labels]

    with open(os.path.join(labels_dir, 'test.txt'), 'w') as f:
        for image in images:
            f.write('{}\n'.format(image))

    participant_dirs = ([_.split('/')[0] for _ in images])
    splits = np.array_split(np.array(participant_dirs), 4)

    for split_idx, split in enumerate(splits):
        split = set(split)
        with open(os.path.join(labels_dir, 'test_{}.txt'.format(split_idx)), 'w') as f:
            for image in images:
                if image.split('/')[0] in split:
                    f.write('{}\n'.format(image))


if __name__ == '__main__':
    main('/data/openeds/openEDS2020-SparseSegmentation/participant')
    # main('/path/to/labels.txt-dir',
    #      'labels.txt')
