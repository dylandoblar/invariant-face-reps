import random
from tqdm import tqdm

from model import TemplateModel
from utils import gen_balanced_pairs_from_dataset, scores_to_acc, load_dataset, write_csv


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

    for idx1, idx2 in tqdm(pairs):
        ex1, label1, fname1 = dataset[idx1]
        ex2, label2, fname2 = dataset[idx2]
        output = model.predict(ex1, ex2)
        label = int(label1 == label2)
        num_pairs += 1
        if output == label:
            num_correct += 1
        else:
            wrong_pairs.append((fname1, label1, fname2, label2))

    accuracy = num_correct / num_pairs
    print(f"[evaluate] num_correct : {num_correct}")
    print(f"[evaluate] num_pairs : {num_pairs}")
    print(f"[evaluate] accuracy : {accuracy}")
    print(f"wrong_pairs :")
    print(wrong_pairs)
    num_false_negatives = {sum([a != b for _, a,_, b in wrong_pairs])}
    num_false_positives = {sum([a == b for _, a,_, b in wrong_pairs])}
    print(f'number wrong with same label : {num_false_negatives}')
    print(f'number wrong with diff label : {num_false_positives}')

    if logging_dir is not None:
        write_csv(wrong_pairs, ["fname1", "label1", "fname2", "label2"],logging_dir+"wrong_pairs.csv")

    return accuracy


if __name__ == '__main__':
    random.seed(1612)

    data = 'extreme'  # 'extreme'  # extreme or normal dataset
    print(f'Experiment type : {data} illumination')

    template_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{data}_template/img'
    model = TemplateModel(
        template_dir,
        repr_type='HOG',
        # repr_type='RANDOM',  # sanity check that with large sample sizes accuracy is nearly 0.5
        pca_dim=100,
        standardize=True,
        num_thresh_samples=100,
        thresh=0.9999560161509551,  # pass in a threshold if you don't want to tune
        num_template_ids=10,
        num_template_samples_per_id=30,
    )

    test_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{data}_test/img'
    # dataset = load_dataset(template_dir)  # sanity check: model does well on the template set
    dataset = load_dataset(test_dir,keep_file_names=True)
    acc = evaluate(model, dataset, num_samples=100)
    print(f"model accuracy on the balanced test set : {acc}")
