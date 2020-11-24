import os
import random
from skimage import io


def load_dataset(dataset_dir, num_ids=0, num_samples_per_id=0, shuffle=False):
    '''
    Loads a dataset from a directory with structure:
        dataset_dir
            identity1
                id1ex1.png
                id1ex2.png
                ...
            ...
            identityN
                idNex1.png
                idNex2.png
                ...
    and returns an iterator of (X, y) tuples where X is an array of pixels and y is the ID.

    dataset_dir(str): path to root of dataset
    num_ids(int): number of IDs to subsample (if 0, use all IDs in the dataset)
    num_samples_per_id(int): number of samples to take for each ID (if 0, use all samples for IDs)
    '''
    labels = os.listdir(dataset_dir)
    if num_ids > 0:
        labels = random.sample(labels, num_ids)
    example_fname_label_pairs = [
        (os.path.join(dataset_dir, label, fname), label)
        for label in labels
        for fname in (os.listdir(os.path.join(dataset_dir, label)) if num_samples_per_id == 0 else
                      random.sample(os.listdir(os.path.join(dataset_dir, label)),
                                    num_samples_per_id))
    ]
    img_label_pairs = [(io.imread(fname), label) for fname, label in example_fname_label_pairs]
    if shuffle:
        random.shuffle(img_label_pairs)  # shuffle in place
    return img_label_pairs


def gen_balanced_pairs_from_dataset(dataset, num_samples=-1):
    '''
    Generates a list of pairs of examples from dataset, with equal number of "same" and "different"
    examples.

    num_samples(int): num_samples//2 pairs will be sampled with (1) the same label and
                      (2) different labels for tuning the threshold. If -1, use as many
                      samples as possible while keeping classes balanced.
    Returns: pairs(list): balanced list of pairs of examples from dataset
    '''
    pairs_same_id = []
    pairs_diff_id = []
    for idx1 in range(len(dataset)):
        _, label1 = dataset[idx1]
        for idx2 in range(idx1+1, len(dataset)):
            _, label2 = dataset[idx2]
            if label1 == label2:
                pairs_same_id.append((idx1, idx2))
            else:
                pairs_diff_id.append((idx1, idx2))
    num_samples_per_label = min(len(pairs_same_id), len(pairs_diff_id))
    if num_samples != -1:
        num_samples_per_label = min(num_samples // 2, num_samples_per_label)
    pairs_same_id = random.sample(pairs_same_id, num_samples_per_label)
    pairs_diff_id = random.sample(pairs_diff_id, num_samples_per_label)
    pairs = pairs_same_id + pairs_diff_id
    return pairs


def scores_to_acc(score_label_pairs, thresh):
    '''
    Given pairs of (score, label) where score is a cosine similarity and label is 0 or 1 ground
    truth, compute the accuracy on the pairs with cutoff threshold thresh (predict 1 for a pair
    if score > thresh).
    '''
    return sum([int(int(score > thresh) == label) for score, label in score_label_pairs]) /\
        len(score_label_pairs)
