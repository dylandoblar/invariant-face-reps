import random
from tqdm import tqdm

from model import TemplateModel
from utils import *
from sklearn.metrics import classification_report,f1_score,matthews_corrcoef,auc

def evaluate(model, dataset, num_samples=-1,logging_dir=None):
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
    ypred= []
    ytrue= []

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
        else: wrong_pairs.append((fname1, label1, fname2, label2))

        sampled_identies.update({label1, label2})

    num_correct = np.sum([1 for pred,gt in zip(ypred,ytrue) if pred == gt])
    num_pairs = len(ytrue)
    accuracy = num_correct / num_pairs

    (fp, tn, fn, tp) = get_fp_tn_fn_tp(ytrue, ypred)
    fpr, tpr = get_fpr_tpr(ytrue, ypred)
    metric_map = {"accuracy": accuracy,
        "mcc": matthews_corrcoef(ytrue, ypred),
        "precision":tp/(tp+fp),
        "recall": tp / (tp + fn),
        "fpr": fpr,
        "tpr": tpr,
        "f1_score":f1_score(ytrue, ypred)}

    # print(f"[evaluate] num_correct : {num_correct}")
    # print(f"[evaluate] num_pairs : {num_pairs}")
    print(f"[evaluate] accuracy : {accuracy}")
    print(f'number wrong with same label : {fn}')
    print(f'number wrong with diff label : {fp}')
    print(classification_report(ytrue,ypred))

    if logging_dir is not None:
        if not os.path.exists(logging_dir): os.makedirs(logging_dir)
        write_csv(wrong_pairs, ["fname1", "label1", "fname2", "label2"],logging_dir+"wrong_pairs.csv")
        write_csv(correct_pairs, ["fname1", "label1", "fname2", "label2"],logging_dir+"correct_pairs.csv")
        analyze_errors(wrong_pairs, correct_pairs, sampled_identies, logging_dir)
        roc_data = plot_roc(get_model_scores(model, dataset, pairs),logging_dir+"test_threshold_roc.png")
        save_data(roc_data, logging_dir + "roc_data.json")
        metric_map["auc"] = roc_data["auc_score"]
        save_data(metric_map, logging_dir+"metrics.json")


    return accuracy

def run_experiment(repr_type, template_data, test_data, vgg_model_type=None, num_ids=50, num_per_id=15, num_pca_dim=50,sample_complexity_exp=False):
    random.seed(1612)

    # manage data + logging directories
    template_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{template_data}_mvn_template/img'
    # if vgg_model_type is not None:logging_dir = f'./logging_dir/{repr_type}_{data}_{test_data}_{vgg_model_type}/'
    # else: logging_dir = f'./logging_dir/{repr_type}_{data}_{test_data}/'

    # change data saving if running sample complexity exp
    if sample_complexity_exp:
        logging_dir = f'./logging_dir/sample_complexity/{repr_type}_{template_data}_{test_data}_{vgg_model_type}_{num_ids}_{num_per_id}/'
    else: logging_dir = f'./logging_dir/{repr_type}_{template_data}_{test_data}_{vgg_model_type}/'

    if not os.path.exists(logging_dir):os.makedirs(logging_dir)
    else: return logging_dir  # don't re-run experiment if directory exists (change if need to rerun)

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
    # model = TemplateModel(
    #     template_dir,
    #     repr_type=repr_type,
    #     pca_dim=25,
    #     standardize=True,
    #     num_thresh_samples=50,
    #     num_template_ids=5,
    #     num_template_samples_per_id=5,
    #     vgg_model_path=f'vgg_model_{vgg_model_type}.h5',
    #     logging_dir=logging_dir,
    # )

    test_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{test_data}_mvn_test/img'
    # dataset = load_dataset(template_dir)  # sanity check: model does well on the template set
    dataset = load_dataset(test_dir,keep_file_names=True)
    acc = evaluate(model, dataset, num_samples=100,logging_dir=logging_dir)
    print(f"model accuracy on the balanced test set : {acc}")

    # num_ids = 10
    # data_subset = load_dataset(test_dir, num_ids=num_ids, num_samples_per_id=0, shuffle=True,keep_file_names=True)
    # emb_data = compute_tsne(model, data_subset, logging_dir + "tnse.png", num_classes=num_ids)

    return logging_dir

def run_tsne(repr_type, template_data, test_data, vgg_model_type=None, num_ids=50, num_per_id=15, num_pca_dim=50):
    random.seed(1612)

    # manage data + logging directories
    template_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{template_data}_mvn_template/img'
    # if vgg_model_type is not None:logging_dir = f'./logging_dir/{repr_type}_{data}_{test_data}_{vgg_model_type}/'
    # else: logging_dir = f'./logging_dir/{repr_type}_{data}_{test_data}/'

    # change data saving if running sample complexity exp
    logging_dir = f'./logging_dir/tsne/{repr_type}_{template_data}_{test_data}_{vgg_model_type}/'

    if not os.path.exists(logging_dir):os.makedirs(logging_dir)
    #else: return logging_dir  # don't re-run experiment if directory exists (change if need to rerun)

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

    test_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{test_data}_mvn_test/img'
    num_ids = 10
    data_subset = load_dataset(test_dir, num_ids=num_ids, num_samples_per_id=0, shuffle=True,keep_file_names=True)
    title = f'{repr_type}-Model Features: {template_data} Templates, {test_data} Test'
    emb_data = compute_tsne(model, data_subset, logging_dir, title, use_raw_features = False, num_classes=num_ids)
    title = f'{repr_type} Raw Features: {template_data} Templates, {test_data} Test'
    emb_data = compute_tsne(model, data_subset, logging_dir, title, use_raw_features = True, num_classes=num_ids)



    return logging_dir

if __name__ == '__main__':
    # random.seed(1612)

    final_output_dir = "./final_plots/"
    if not os.path.exists(final_output_dir): os.makedirs(final_output_dir)

    repr_types = ["HOG", "VGG"]
    vgg_model_types = ["extreme", "normal", "pretrain"]
    poss_templates = ["normal", "extreme"]
    poss_tests = ["normal", "extreme"]

    template_data = "normal"
    test_data = "extreme"
    vgg_model_type = "extreme"

    logging_dirs = []
    for repr_type in repr_types:
        logging_dir = run_tsne(repr_type, template_data, test_data, vgg_model_type)

    # run sample complexity exp (num_ids, num_per_id)
    # settings = [(15,50), (30,25), (25,30), (50,15), (150,5),(250,3)]
    # for repr_type in repr_types:
    #     labels = []
    #     logging_dirs = []
    #     for num_ids, num_per_id in settings:
    #         print("running: ", num_ids, num_per_id)
    #         logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type, num_ids, num_per_id, True)
    #         logging_dirs.append(logging_dir)
    #         labels.append(f'Num IDs: {num_ids}, Num Per ID: {num_per_id}')
    #     styles = ['b-*', 'r-o', 'g--', 'p-*', 'm-o', 'c--', 'y-*']
    #     file_tag = f'./final_plots/sample_complexity_{repr_type}_{template_data}_{test_data}.png'
    #     title = f'Sample Complexity: {repr_type} on Novel Extreme Imgs'
    #     create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)


# if __name__ == '__main__':
#     #random.seed(1612)
#
#     final_output_dir = "./final_plots/"
#     if not os.path.exists(final_output_dir): os.makedirs(final_output_dir)
#
#     repr_types = ["HOG", "VGG"]
#     vgg_model_types = ["extreme", "normal", "pretrain"]
#     poss_templates = ["normal", "extreme"]
#     poss_tests = ["normal", "extreme"]
#
#     template_data = "normal"
#     test_data = "extreme"
#     vgg_model_type = "extreme"
#
#     # run sample complexity exp (num_ids, num_per_id)
#     # settings = [(15,50), (30,25), (25,30), (50,15), (150,5),(250,3)]
#     # for repr_type in repr_types:
#     #     labels = []
#     #     logging_dirs = []
#     #     for num_ids, num_per_id in settings:
#     #         print("running: ", num_ids, num_per_id)
#     #         logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type, num_ids, num_per_id, True)
#     #         logging_dirs.append(logging_dir)
#     #         labels.append(f'Num IDs: {num_ids}, Num Per ID: {num_per_id}')
#     #     styles = ['b-*', 'r-o', 'g--', 'p-*', 'm-o', 'c--', 'y-*']
#     #     file_tag = f'./final_plots/sample_complexity_{repr_type}_{template_data}_{test_data}.png'
#     #     title = f'Sample Complexity: {repr_type} on Novel Extreme Imgs'
#     #     create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)

    # all_num_ids = [5, 10, 25, 50, 100]#500]
    # all_samp_per_id = [5,10,15,25,50]
    # num_pca_dim = 25
    # for repr_type in repr_types:
    #     logging_dirs = []
    #     for num_ids in all_num_ids:
    #         for num_samp_per_id in all_samp_per_id:
    #             print("running: ", num_ids, num_samp_per_id)
    #             logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type, num_ids, num_samp_per_id, num_pca_dim,
    #                                          True)
    #             logging_dirs.append(logging_dir)
    #     for metric in ["mcc", "auc", "accuracy", "f1_score"]:
    #         file_tag = f'./final_plots/{metric}_sample_complexity_heatmap_{repr_type}_{template_data}_{test_data}.png'
    #         title = f'Sample Complexity: {repr_type} on Novel Extreme Imgs, {metric}'
    #         complexity_heatmap(logging_dirs, all_num_ids, all_samp_per_id, title, file_tag, metric)

    # template_data = "extreme"
    # test_data = "extreme"
    # vgg_model_type = "extreme"
    #
    # all_num_ids = [5, 10, 25, 50, 100]#500]
    # all_samp_per_id = [5,10,15,25,50]
    # num_pca_dim = 25
    # for repr_type in repr_types:
    #     logging_dirs = []
    #     for num_ids in all_num_ids:
    #         for num_samp_per_id in all_samp_per_id:
    #             print("running: ", num_ids, num_samp_per_id)
    #             logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type, num_ids, num_samp_per_id, num_pca_dim,
    #                                          True)
    #             logging_dirs.append(logging_dir)
    #     for metric in ["mcc", "auc", "accuracy", "f1_score"]:
    #         file_tag = f'./final_plots/{metric}_sample_complexity_heatmap_{repr_type}_{template_data}_{test_data}.png'
    #         title = f'Sample Complexity: {repr_type} on Extreme/Extreme, {metric}'
    #         complexity_heatmap(logging_dirs, all_num_ids, all_samp_per_id, title, file_tag, metric)




    # logging_dirs = []
    # logging_dirs.append(run_experiment("HOG", template_data, test_data, "normal")) # vgg model type doesn't matter for hog (ignore)
    # repr_type = "VGG"
    # for vgg_model_type in vgg_model_types:
    #     logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type)
    #     logging_dirs.append(logging_dir)
    #
    # labels = ["HOG Features", "VGG, Fine-Tuned (Extreme)", "VGG, Fine-Tuned (Normal)", "VGG, Pretrain Only"]
    # styles = ['b-*', 'r-o', 'g--', 'p-*']
    # file_tag = "compare_normal_extreme_all.png"
    # title = "ROC Plot: Normal Templates, Extreme Test Set"
    # create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)

#
# if __name__ == '__main__':
#     random.seed(1612)
#
#     final_output_dir = "./final_plots/"
#     if not os.path.exists(final_output_dir): os.makedirs(final_output_dir)
#
#     repr_types = ["HOG", "VGG"]
#     vgg_model_types = ["normal", "extreme", "pretrain"]
#     poss_templates = ["normal", "extreme"]
#     poss_tests = ["normal", "extreme"]
#
#     # # 1) compare HOG vs. VGG on normal illumination test set
#     # template_data = "normal"
#     # test_data = "normal"
#     # vgg_model_type = "normal"
#     #
#     # logging_dirs = []
#     # for repr_type in repr_types:
#     #     logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type)
#     #     logging_dirs.append(logging_dir)
#     #
#     # title = "ROC Plot: Test on Normal Illumination"
#     # labels = ["HOG", "VGG"]
#     # file_tag = "normal_illum_test.png"
#     # styles = ['b-*', 'r-o']
#     # create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)
#     #
#     # # 2) compare VGG training procedure for extreme illumination test set (extreme input)
#     #
#     # repr_type = "VGG"
#     # template_data = "extreme"
#     # test_data = "extreme"
#     #
#     # logging_dirs = []
#     # for vgg_model_type in vgg_model_types:
#     #     logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type)
#     #     logging_dirs.append(logging_dir)
#     #
#     # title = "ROC Plot: Vary VGGFace Training Data (Extreme Templates, Extreme Test)"
#     # labels = ["Fine-Tune Normal", "Fine-Tune Extreme", "Pretrain Only"]
#     # file_tag = "vgg_model_test_extreme.png"
#     # styles = ['b-*', 'r-o', 'g--']
#     # create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)
#     #
#     # # 3) compare VGG training procedure for extreme illumination test set (normal input)
#     #
#     # repr_type = "VGG"
#     # template_data = "normal"
#     # test_data = "extreme"
#     #
#     # logging_dirs = []
#     # for vgg_model_type in vgg_model_types:
#     #     logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type)
#     #     logging_dirs.append(logging_dir)
#     #
#     # title = "ROC Plot: Vary VGGFace Training Data (Normal Templates, Extreme Test)"
#     # labels = ["Fine-Tune Normal", "Fine-Tune Extreme", "Pretrain Only"]
#     # file_tag = "vgg_model_test_extreme_normal_template.png"
#     # styles = ['b-*', 'r-o', 'g--']
#     # create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)
#
#     # 4) check VGG-extreme on all settings
#     repr_type = "VGG"
#     vgg_model_type = "extreme"
#     labels = []
#     logging_dirs = []
#     for template_data in poss_templates:
#         for test_data in poss_tests:
#             logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type)
#             logging_dirs.append(logging_dir)
#
#     labels = ["Normal Template, Normal Test", "Normal Template, Extreme Test", "Extreme Template, Normal Test", "Extreme Template, Extreme Test"]
#     styles = ['b-*', 'r-o', 'g--', 'p-*']
#     file_tag = "vgg_extreme_all.png"
#     title = "ROC Plot: VGG-Extreme vs. Template/Test Pairs"
#     create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)
#
#     # 5) check HOG on all settings
#     repr_type = "HOG"
#     labels = []
#     logging_dirs = []
#     for template_data in poss_templates:
#         for test_data in poss_tests:
#             logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type)
#             logging_dirs.append(logging_dir)
#
#     labels = ["Normal Template, Normal Test", "Normal Template, Extreme Test", "Extreme Template, Normal Test", "Extreme Template, Extreme Test"]
#     styles = ['b-*', 'r-o', 'g--', 'p-*']
#     file_tag = "hog_all.png"
#     title = "ROC Plot: HOG vs. Template/Test Pairs"
#     create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)
#
#     # 6) check random projection matrix for all settings of template/test
#
#     repr_type = "RANDOM"
#     labels = []
#     logging_dirs = []
#     for template_data in poss_templates:
#         for test_data in poss_tests:
#             logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type)
#             logging_dirs.append(logging_dir)
#
#     labels = ["Normal Template, Normal Test", "Normal Template, Extreme Test", "Extreme Template, Normal Test", "Extreme Template, Extreme Test"]
#     styles = ['b-*', 'r-o', 'g--', 'p-*']
#     file_tag = "random_all.png"
#     title = "ROC Plot: Random vs. Template/Test Pairs"
#     create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)
#
#     # 7) check VGG-extreme on all settings
#     repr_type = "VGG"
#     vgg_model_type = "normal"
#     labels = []
#     logging_dirs = []
#     for template_data in poss_templates:
#         for test_data in poss_tests:
#             logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type)
#             logging_dirs.append(logging_dir)
#
#     labels = ["Normal Template, Normal Test", "Normal Template, Extreme Test", "Extreme Template, Normal Test", "Extreme Template, Extreme Test"]
#     styles = ['b-*', 'r-o', 'g--', 'p-*']
#     file_tag = "vgg_normal_all.png"
#     title = "ROC Plot: VGG-Normal vs. Template/Test Pairs"
#     create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)
#
#     # 8) check VGG-extreme on all settings
#     repr_type = "VGG"
#     vgg_model_type = "pretrain"
#     labels = []
#     logging_dirs = []
#     for template_data in poss_templates:
#         for test_data in poss_tests:
#             logging_dir = run_experiment(repr_type, template_data, test_data, vgg_model_type)
#             logging_dirs.append(logging_dir)
#
#     labels = ["Normal Template, Normal Test", "Normal Template, Extreme Test", "Extreme Template, Normal Test", "Extreme Template, Extreme Test"]
#     styles = ['b-*', 'r-o', 'g--', 'p-*']
#     file_tag = "vgg_pretrain_all.png"
#     title = "ROC Plot: VGG Pretrain-Only vs. Template/Test Pairs"
#     create_overlayed_rocs(title, labels, styles, logging_dirs, final_output_dir + file_tag)