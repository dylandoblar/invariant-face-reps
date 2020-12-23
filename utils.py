import os
import random
from skimage import io
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import auc
import matplotlib
import shutil

matplotlib.rcParams['font.sans-serif'] = "Computer Modern Sans Serif"
matplotlib.rcParams['font.family'] = "sans-serif"


axis_text_size = 18
title_text_size = 18
legend_text_size = 12


def load_dataset(
    dataset_dir,
    num_ids=0,
    num_samples_per_id=0,
    shuffle=False,
    keep_file_names=False
):
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
    keep_file_names(bool): if True, return file names s.t. (img, label, filename)
    '''
    labels = [label for label in os.listdir(dataset_dir) if label != ".DS_Store"]
    if num_ids > 0:
        labels = random.sample(labels, num_ids)
    example_fname_label_pairs = [
        (os.path.join(dataset_dir, label, fname), label)
        for label in labels
        for fname in (os.listdir(os.path.join(dataset_dir, label)) if num_samples_per_id == 0 else
                      random.sample(os.listdir(os.path.join(dataset_dir, label)),
                                    num_samples_per_id))
    ]
    if keep_file_names:
        img_label_pairs = [
            (io.imread(fname), label, fname) for fname, label in example_fname_label_pairs
        ]
    else:
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
        label1 = dataset[idx1][1]  # label is always idx 1 in datum tuple
        for idx2 in range(idx1+1, len(dataset)):
            label2 = dataset[idx2][1]
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


def write_csv(data, column_labels, file_path):
    '''
    Take in list of tuples and associated list of column labels
    Save as a .csv
    '''
    with open(file_path, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(column_labels)
        csv_out.writerows(data)


def get_illum_setting(filename):
    '''
    Takes in a file name (.png) and returns the illumination setting as int
    '''
    img_name = filename.split("/")[-1]
    illum_setting = img_name.split("_")[1].split(".")[0]
    return int(illum_setting)


def analyze_errors(wrong_preds, correct_preds, sampled_identies, file_path, num_settings=50):
    '''
    format of preds = (filename1, predlabel1, filename2, predlabel2)
    takes in number of illumination settings (optional)
    '''
    poss_ids = sorted(list(sampled_identies))
    num_ids = len(poss_ids)
    id_map = {id: idx for idx, id in enumerate(poss_ids)}

    identity_errors = np.zeros([num_ids, num_ids])
    illum_cond_errors = np.zeros([num_settings, num_settings])

    for f1, l1, f2, l2 in wrong_preds:
        identity_errors[id_map[l1], id_map[l2]] += 1
        illum_cond_errors[get_illum_setting(f1), get_illum_setting(f2)] += 1

    plt.figure()
    sns.heatmap(identity_errors)  # , xticklabels=poss_ids, yticklabels=poss_ids)
    plt.savefig(file_path+"wrong_ids.png")

    plt.figure()
    sns.heatmap(illum_cond_errors)
    plt.savefig(file_path + "wrong_illum_cond.png")


def get_fp_tn_fn_tp(ytrue, ypred):
    '''
    Compute various classification metrics
    Take as input lists of binary (0,1) labels of equal length
    0 = diff id, 1 = same id (positive = same id = 1)
    Return as tuple: (fp, tn, fn, tp)
    '''
    fp = np.sum([1 for pred, gt in zip(ypred, ytrue) if pred == 1 and gt == 0])
    tn = np.sum([1 for pred, gt in zip(ypred, ytrue) if pred == 0 and gt == 0])
    fn = np.sum([1 for pred, gt in zip(ypred, ytrue) if pred == 0 and gt == 1])
    tp = np.sum([1 for pred, gt in zip(ypred, ytrue) if pred == 1 and gt == 1])
    return (fp, tn, fn, tp)


def get_fpr_tpr(ytrue, ypred):
    '''
    Compute FPR and TPR
    Take as input lists of binary (0,1) labels of equal length
    0 = diff id, 1 = same id (positive = same id = 1)
    Return as tuple: (fpr, tpr)
    '''
    (fp, tn, fn, tp) = get_fp_tn_fn_tp(ytrue, ypred)

    fpr = fp/(fp + tn)
    tpr = tp/(tp + fn)

    return (fpr, tpr)


def get_fpr_tpr_thresh(score_label_pairs, thresh):
    '''
    Compute FPR and TPR
    Take as input pairs of (score, label) where score is a cosine similarity
        and label is 0 or 1 ground truth.
    Extract pred list as if score > thresh = 1
    Return as tuple: (fpr, tpr)
    '''

    ypred = [int(score > thresh) for score, _ in score_label_pairs]
    ytrue = [label for _, label in score_label_pairs]

    return get_fpr_tpr(ytrue, ypred)


def plot_roc(score_label_pairs, output_path):
    '''
    Create ROC plot from metrics (takes in pairs of (fpr, tpr))
    Returns AUC (float)
    '''

    template_scores = list(zip(*score_label_pairs))[0]
    roc_metrics = [get_fpr_tpr_thresh(score_label_pairs, thresh) for thresh in template_scores]

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')  # plt x = y control line
    fprs = [fpr for fpr, _ in roc_metrics]
    tprs = [tpr for _, tpr in roc_metrics]
    auc_score = auc(fprs, tprs)
    plt.plot(fprs, tprs)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve, AUC: {round(auc_score,3)}')
    plt.savefig(output_path)
    return {"auc_score": auc_score, "fprs": fprs, "tprs": tprs}


def plot_many_rocs(all_fprs, all_tprs, labels, styles, output_path, title="ROC curve"):
    '''
    Create k overlayed ROC plots
    Takes as input k-dimensional lists of fprs + tprs, and labels
    Styles are matplotlib color + point type (i.e., g*)
    Save to output_path
    '''

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')  # plt x = y control line

    for label, fprs, tprs, style in zip(labels, all_fprs, all_tprs, styles):
        auc_score = auc(fprs, tprs)
        plt.plot(fprs, tprs, label=f'{label}, AUC: {round(auc_score,3)}')
    plt.xlabel('False positive rate', fontsize=axis_text_size)
    plt.ylabel('True positive rate', fontsize=axis_text_size)
    plt.title(title, fontsize=title_text_size)
    plt.legend(loc="lower right", prop={'size': legend_text_size})
    plt.savefig(output_path)


def create_overlayed_rocs(title, labels, styles, dirs, output_path):
    all_fprs = []
    all_tprs = []
    for logging_dir in dirs:
        # read in fprs and tprs
        with open(logging_dir + "roc_data.json") as f:
            roc_data = json.load(f)
            all_fprs.append(roc_data["fprs"])
            all_tprs.append(roc_data["tprs"])
    plot_many_rocs(all_fprs, all_tprs, labels, styles, output_path, title)


def plot_complexity(
    param_val,
    metrics,
    filepath,
    xlab="Samples",
    ylab="Acc",
    title="Sample Complexity"
):
    '''
    Create plot of sample complexity (x = vary parameters, y = assoc performance metrics)
    '''
    plt.figure()
    plt.plot(param_val, metrics)
    plt.xlabel(xlab, fontsize=axis_text_size)
    plt.ylabel(ylab, fontsize=axis_text_size)
    plt.title(title, fontsize=title_text_size)
    plt.savefig(filepath)


def get_sample_complexity_params(logging_dir):
    # get sample complexity parameters (num ids, num per id) from directory name
    decomp_dir = logging_dir[:-1].split("_")
    return int(decomp_dir[-2]), int(decomp_dir[-1])  # final clauses = num_id + num_per_id


def complexity_heatmap(dirs, all_num_ids, all_samp_per_id, title, output_path, metric="auc"):

    all_performance_metrics = []

    # create map of parameter setting to auc of the logging directory
    param_dir_map = {}

    for logging_dir in dirs:  # format has _numIDs_numPerID/
        num_ids, num_samp_per_id = get_sample_complexity_params(logging_dir)
        # read in fprs and tprs
        with open(logging_dir + "metrics.json") as f:
            metrics_map = json.load(f)
            param_dir_map[(num_ids, num_samp_per_id)] = metrics_map

    all_samp_per_id = sorted(all_samp_per_id, reverse=True)  # so that bottom left is smallest # sam

    for num_ids in all_num_ids:
        performance_metrics = []
        for num_samp_per_id in all_samp_per_id:
            performance_metrics.append(param_dir_map[(num_ids, num_samp_per_id)][metric])
        all_performance_metrics.append(performance_metrics)

    plt.figure()
    sns.heatmap(
        all_performance_metrics,
        cmap='RdBu_r',
        vmin=0,
        vmax=1,
        xticklabels=list(map(str, all_num_ids)),
        yticklabels=list(map(str, all_samp_per_id)),
    )
    if title is not None:
        plt.title(title)
    plt.xlabel("Num IDs", fontsize=axis_text_size)
    plt.ylabel("Num Samples per ID", fontsize=axis_text_size)
    plt.savefig(output_path)


def save_data(data, filepath):
    '''
    Save data as .json file
    '''
    with open(filepath, "w") as f:
        json.dump(data, f)


def get_model_scores(model, dataset, pairs):
    '''
    Gets (score,label) pairs of model evaluated on (ex1,ex2) pairs from dataset
    Score is cosine similarity (pre-threshold)
    '''
    score_label_pairs = []
    for idx1, idx2 in tqdm(pairs):
        ex1, label1, fname1 = dataset[idx1]
        ex2, label2, fname2 = dataset[idx2]
        score = model.score(ex1, ex2)
        label = int(label1 == label2)
        score_label_pairs.append((score, label))
    score_label_pairs.sort(key=lambda x: x[0])
    return score_label_pairs


def plot_tsne(emb_df, plot_path, title=None, legend_type=False, num_classes=10):
    # inspired by: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-\
    #    and-t-sne-in-python-8ef87e7915b
    print("emb df: ", emb_df.head(5))
    plt.figure(figsize=(16, 10))
    emb_df["ID"] = emb_df["id"]  # change title for legend (if added)
    g = sns.scatterplot(
        x="emb1", y="emb2",
        hue="ID",
        palette=sns.color_palette("hls", num_classes),
        data=emb_df,
        legend=legend_type,
        alpha=0.7
    )
    if title is not None:
        plt.title(title, fontsize=40)
    plt.savefig(plot_path)


def compute_tsne(
    model,
    dataset,
    data_dir,
    plot_path,
    title=None,
    legend_type=False,
    use_raw_features=True,
    num_classes=10,
):
    '''
    Project dataset using HOG/VGG features
    Compute and plot TSNE
    Raw features = direct HOG features or VGG activations
    '''

    from MulticoreTSNE import MulticoreTSNE as TSNE

    X = []
    ids = []
    filenames = []
    for (img, label, fname) in dataset:
        if use_raw_features:
            features = model.compute_feats(img)
        else:
            features = model.compute_representation(img)[0]

        X.append(features)
        ids.append(label)
        filenames.append(fname)
    X = np.array(X)
    print("data matrix: ", np.shape(X))

    # compute embeddings
    print("starting tsne.....")
    embeddings = TSNE(n_jobs=4, random_state=7).fit_transform(X)
    print("done with tsne")

    # inspired by: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-\
    #    and-t-sne-in-python-8ef87e7915b
    emb_df = pd.DataFrame(
        data={"emb1": embeddings[:, 0], "emb2": embeddings[:, 1], "id": ids, "filename": filenames}
    )

    if use_raw_features:
        data_dir += "raw_features_"
    else:
        data_dir += "model_rep_"
    emb_df.to_csv(data_dir + "tsne_data.csv")

    plot_tsne(emb_df, plot_path, title, legend_type, num_classes)

    return embeddings


def split_dataset(src, split_size=0.5):
    random.seed(1612)

    template_dir = src + '_template/img/'
    test_dir = src + '_test/img/'
    all_files = np.array(os.listdir(src))

    np.random.shuffle(all_files)
    template_files, test_files = np.split(
        np.array(all_files),
        [int(len(all_files) * (1 - split_size))]
    )

    template_files = [name for name in template_files.tolist() if name != ".DS_Store"]
    test_files = [name for name in test_files.tolist() if name != ".DS_Store"]

    print('Total images: ', len(all_files))
    print('Training: ', len(template_files))
    print('Testing: ', len(test_files))

    for f in template_files:
        if f == ".DS_Store":
            continue
        local_root = os.path.join(src, f)
        for idx, orig_name in enumerate(os.listdir(local_root)):
            if orig_name == ".DS_Store":
                continue
            new_dir = template_dir + f + "/"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            shutil.copy2(local_root + "/" + orig_name, new_dir + f + "_" + str(idx) + ".jpg")

    for f in test_files:
        if f == ".DS_Store":
            continue
        local_root = os.path.join(src, f)
        for idx, orig_name in enumerate(os.listdir(local_root)):
            if orig_name == ".DS_Store":
                continue
            new_dir = test_dir + f + "/"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            shutil.copy2(local_root + "/" + orig_name, new_dir + f + "_" + str(idx) + ".jpg")

    return template_dir, test_dir
