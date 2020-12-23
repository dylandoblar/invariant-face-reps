import random
from tqdm import tqdm

from model import TemplateModel
from utils import *
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, auc
import shutil

# NOTE(katie): run "source env/bin/activate" to start this virtualenv


def evaluate(model, dataset, num_samples=-1, logging_dir=None):
    '''
    Evaluates the model on pairs of examples in the dataset specified by dataset_path.
    Returns accuracy on sampled pairs of examples in the dataset. Classes are balanced such
    that there are equal number of same and different pairs in the evaluation set.
    num_samples(int): num_samples//2 pairs will be sampled with (1) the same label and
                        (2) different labels for tuning the threshold. If -1, use as many
                        samples as possible while keeping classes balanced.
    logging_dir(str): where to save metric + analysis data, if desired
    '''
    print('Evaluating model:')
    pairs = gen_balanced_pairs_from_dataset(dataset, num_samples)

    num_correct = 0
    num_pairs = 0
    wrong_pairs = []
    correct_pairs = []
    sampled_identies = set()
    ypred = []
    ytrue = []

    for idx1, idx2 in tqdm(pairs):
        ex1, label1, fname1 = dataset[idx1]
        ex2, label2, fname2 = dataset[idx2]
        output = model.predict(ex1, ex2)
        label = int(label1 == label2)
        ypred.append(output)
        ytrue.append(label)
        num_pairs += 1
        if output == label:
            num_correct += 1
            correct_pairs.append((fname1, label1, fname2, label2))
        else:
            wrong_pairs.append((fname1, label1, fname2, label2))

        sampled_identies.update({label1, label2})

    num_correct = np.sum([1 for pred, gt in zip(ypred, ytrue) if pred == gt])
    num_pairs = len(ytrue)
    accuracy = num_correct / num_pairs

    (fp, tn, fn, tp) = get_fp_tn_fn_tp(ytrue, ypred)
    fpr, tpr = get_fpr_tpr(ytrue, ypred)
    metric_map = {
        "accuracy": accuracy,
        "mcc": matthews_corrcoef(ytrue, ypred),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "fpr": fpr,
        "tpr": tpr,
        "f1_score": f1_score(ytrue, ypred)
    }

    # print(f"[evaluate] num_correct : {num_correct}")
    # print(f"[evaluate] num_pairs : {num_pairs}")
    print(f"[evaluate] accuracy : {accuracy}")
    print(f'number wrong with same label : {fn}')
    print(f'number wrong with diff label : {fp}')
    print(classification_report(ytrue, ypred))

    if logging_dir is not None:
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        write_csv(
            wrong_pairs,
            ["fname1", "label1", "fname2", "label2"],
            logging_dir + "wrong_pairs.csv"
        )
        write_csv(
            correct_pairs,
            ["fname1", "label1", "fname2", "label2"],
            logging_dir + "correct_pairs.csv"
        )
        # analyze_errors(wrong_pairs, correct_pairs, sampled_identies, logging_dir)
        roc_data = plot_roc(
            get_model_scores(model, dataset, pairs),
            logging_dir+"test_threshold_roc.png"
        )
        save_data(roc_data, logging_dir + "roc_data.json")
        metric_map["auc"] = roc_data["auc_score"]
        save_data(metric_map, logging_dir + "metrics.json")

    return accuracy


def run_tsne(
    titles,
    plot_output_dir,
    repr_type,
    template_data,
    test_data,
    vgg_model_type=None,
    num_ids=50,
    num_per_id=15,
    num_pca_dim=50,
    num_classes=10
):
    # random.seed(1612)
    # manage data + logging directories
    if template_data == "pubfig83":
        template_dir = f'/Users/kcollins/invariant_face_data/{template_data}/template'
        # change data saving if running sample complexity exp
        logging_dir = f'./logging_dir/{data_type}/{repr_type}_{vgg_model_type}/'
        test_dir = f'/Users/kcollins/invariant_face_data/{template_data}/test'
        legend_type = "full"
    else:
        template_dir = \
            f'/Users/kcollins/invariant_face_data/illum_data/ill_{template_data}_mvn_template/img'
        # change data saving if running sample complexity exp
        logging_dir = \
            f'./logging_dir/tsne/{repr_type}_{template_data}_{test_data}_{vgg_model_type}/'
        test_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{test_data}_mvn_test/img'

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # check if embedding data is already saved
    data_file = logging_dir + "model_rep_tsne_data.csv"
    if os.path.exists(data_file):
        # just get plots
        # read in embedding data
        emb_df = pd.read_csv(data_file)
        filepath = \
            f'{plot_output_dir}{repr_type}_{template_data}_{test_data}_model_features_tsne.png'
        plot_tsne(emb_df, filepath, titles[0], legend_type, num_classes)

        data_file = logging_dir + "raw_features_tsne_data.csv"
        emb_df = pd.read_csv(data_file)
        filepath = f'{plot_output_dir}{repr_type}_{template_data}_{test_data}_raw_features_tsne.png'
        plot_tsne(emb_df, filepath, titles[1], legend_type, num_classes)
        return logging_dir
    else:
        # otherwise, run everything:

        # create model
        model = TemplateModel(
            template_dir,
            repr_type=repr_type,
            pca_dim=num_pca_dim,
            standardize=True,
            num_thresh_samples=500,
            num_template_ids=num_ids,
            num_template_samples_per_id=num_per_id,
            vgg_model_path=f'vgg_model_{vgg_model_type}.h5',
            logging_dir=logging_dir,
        )

        data_subset = load_dataset(
            test_dir,
            num_ids=num_classes,
            num_samples_per_id=0,
            shuffle=True,
            keep_file_names=True
        )
        filepath = \
            f'{plot_output_dir}{repr_type}_{template_data}_{test_data}_model_features_tsne.png'
        emb_data = compute_tsne(
            model,
            data_subset,
            logging_dir,
            filepath,
            title,
            use_raw_features=False,
            num_classes=num_classes
        )
        filepath = \
            f'{plot_output_dir}{repr_type}_{template_data}_{test_data}_raw_features_tsne.png'
        emb_data = compute_tsne(
            model,
            data_subset,
            logging_dir,
            filepath,
            title,
            use_raw_features=True,
            num_classes=num_classes
        )
    return logging_dir


def run_experiment(
    repr_type,
    template_dir,
    test_dir,
    logging_dir,
    vgg_model_type=None,
    num_ids=50,
    num_per_id=15,
    num_pca_dim=50,
    num_thresh=500,
):
    # random.seed(7)

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    # else:
    #     print("skipping: ", logging_dir)
    #     return logging_dir  # don't re-run experiment if directory exists (change if need rerun)

    # create model
    model = TemplateModel(
        template_dir,
        repr_type=repr_type,
        pca_dim=num_pca_dim,
        standardize=True,
        num_thresh_samples=num_thresh,
        num_template_ids=num_ids,
        num_template_samples_per_id=num_per_id,
        vgg_model_path=f'vgg_model_{vgg_model_type}.h5',
        logging_dir=logging_dir,
    )

    dataset = load_dataset(test_dir, keep_file_names=True)
    acc = evaluate(model, dataset, num_samples=100, logging_dir=logging_dir)
    print(f"model accuracy on the balanced test set : {acc}")

    return logging_dir


if __name__ == '__main__':

    final_output_dir = "./vss_plots/"
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    do_tsne = False

    main_logging_dir = f'./logging_dir/'
    data_dir = f'/Users/kcollins/invariant_face_data/illum_data/'
    styles = ['b-*', 'r-o', 'g--', 'p-*', 'm-o', 'c--', 'y-*', 'r-^', 'k-o', 'g-*']
    all_metrics = ["mcc", "auc", "accuracy", "f1_score"]

    repr_types = ["HOG", "VGG"]
    vgg_model_type = ["face", "imagenet"]
    poss_templates = ["normal", "extreme"]
    poss_tests = ["normal", "extreme"]

    num_ids = 50
    num_per_id = 15
    num_pca_dim = 50
    num_thresh = 500

    vgg_model_type = "face"
    test_data = "extreme"
    logging_dirs = []
    for template_data in poss_templates:
        template_dir = \
            f'/Users/kcollins/invariant_face_data/illum_data/ill_{template_data}_mvn_template/img'
        test_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{test_data}_mvn_test/img'
        for repr_type in repr_types:
            logging_dir = \
                f'{main_logging_dir}_{repr_type}_{template_data}_{test_data}_{vgg_model_type}'
            run_experiment(
                repr_type,
                template_dir,
                test_dir,
                logging_dir,
                vgg_model_type,
                num_ids=num_ids,
                num_per_id=num_per_id,
                num_pca_dim=num_pca_dim,
                num_thresh=num_thresh
            )
            logging_dirs.append(logging_dir)
    file_tag = f'hog_vgg_all_{test_data}.png'
    title = f'HOG vs. VGG-Face: Extreme Illumination'
    labels = ["HOG (Natural)", "VGG-Face (Natural)", "HOG (Extreme)", "VGG-Face (Extreme)"]
    create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)
    if do_tsne:
        tsne_title = [None, None]  # depending on presentation mode, don't add title
        run_tsne(tsne_title, final_output_dir, "HOG", "normal", test_data, vgg_model_type)
        run_tsne(tsne_title, final_output_dir, "VGG", "normal", test_data, vgg_model_type)
        run_tsne(tsne_title, final_output_dir, "HOG", "extreme", test_data,
                 vgg_model_type)
        run_tsne(tsne_title, final_output_dir, "VGG", "extreme", test_data,
                 vgg_model_type)
