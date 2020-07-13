import os
import numpy as np
from sklearn.model_selection import KFold


def main(labels_dir, labels_file, num_folds=10, random_state=42):

    with open(os.path.join(labels_dir, labels_file)) as f:
        labels = [_.strip() for _ in f.readlines()]

    participants = {label.split('/')[0] for label in labels}
    participant_names = np.array(sorted(list(participants)))
    kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

    for fold_idx, indices in enumerate(kf.split(participant_names)):
        train_index, val_index = indices

        # Repetitive code, but putting into function becomes even uglier

        train_names = set(participant_names[train_index])
        val_names = set(participant_names[val_index])

        train_file_name = os.path.join(labels_dir, 'fold_{}_{}.txt'.format(fold_idx, 'train'))
        val_file_name = os.path.join(labels_dir, 'fold_{}_{}.txt'.format(fold_idx, 'val'))

        train_labels = [_ for _ in labels if _.split('/')[0] in train_names]
        val_labels = [_ for _ in labels if _.split('/')[0] in val_names]

        with open(train_file_name, 'w') as f:
            for label in train_labels:
                f.write('{}\n'.format(label))

        with open(val_file_name, 'w') as f:
            for label in val_labels:
                f.write('{}\n'.format(label))


if __name__ == '__main__':
    main('/path/to/labels.txt-dir',
         'labels.txt')
