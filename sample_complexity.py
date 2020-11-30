import random
from tqdm import tqdm

from model import TemplateModel
from utils import *
from eval import evaluate

def explore_num_id(data,repr_type, num_ids, num_samp_per_id=30):

    performance_metrics = []
    for n_id in num_ids:
        template_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{data}_template/img'
        logging_dir = f'./logging_dir/{repr_type}_{data}/'
        model = TemplateModel(
            template_dir,
            repr_type=repr_type,
            # repr_type='RANDOM',  # sanity check that with large sample sizes accuracy is nearly 0.5
            pca_dim=50,
            standardize=True,
            num_thresh_samples=int((num_samp_per_id * n_id) / 2),
            num_template_ids=int(n_id),
            num_template_samples_per_id=int(num_samp_per_id),
            vgg_model_path=f"vgg_model_finetune_{data}.h5",
            logging_dir=logging_dir,
        )

        test_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{data}_test/img'
        dataset = load_dataset(test_dir, keep_file_names=True)
        acc = evaluate(model, dataset, num_samples=100, logging_dir=logging_dir)
        performance_metrics.append(acc)

    plot_complexity(num_ids, performance_metrics, logging_dir+"sample_complexity_vary_id.png", "Num IDs", "Accuracy")


def explore_num_sample_per_id(data,repr_type, num_samples_per_id, num_ids=10):
    performance_metrics = []
    for num_samp_per_id in num_samples_per_id:
        template_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{data}_template/img'
        logging_dir = f'./logging_dir/{repr_type}_{data}/'
        model = TemplateModel(
            template_dir,
            repr_type=repr_type,
            # repr_type='RANDOM',  # sanity check that with large sample sizes accuracy is nearly 0.5
            pca_dim=50,
            standardize=True,
            num_thresh_samples=int((num_samp_per_id * num_ids) / 2),
            num_template_ids=int(num_ids),
            num_template_samples_per_id=int(num_samp_per_id),
            vgg_model_path=f"vgg_model_finetune_{data}.h5",
            logging_dir=logging_dir,
        )

        test_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{data}_test/img'
        dataset = load_dataset(test_dir, keep_file_names=True)
        acc = evaluate(model, dataset, num_samples=100, logging_dir=logging_dir)
        performance_metrics.append(acc)

    plot_complexity(num_samples_per_id, performance_metrics, logging_dir + "sample_complexity_vary_samples_per_id.png", "Samples per ID", "Accuracy")

def plot_complexity(param_val,metrics,filepath,xlab="Samples", ylab="Acc", title="Sample Complexity"):
    '''
    Create plot of sample complexity (x = vary parameters, y = assoc performance metrics)
    '''
    plt.figure()
    plt.plot(param_val, metrics)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(filepath)

def score_setting(data,repr_type, num_ids, num_samp_per_id, logging_dir):
    '''
    Run model with specific number of ids + sample
    Apply to same test set each time (based on data type)
    Return accuracy
    '''

    template_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{data}_template/img'
    model = TemplateModel(
        template_dir,
        repr_type=repr_type,
        pca_dim=25,
        standardize=True,
        num_thresh_samples=int((num_samp_per_id * num_ids) / 2),
        num_template_ids=int(num_ids),
        num_template_samples_per_id=int(num_samp_per_id),
        vgg_model_path=f"vgg_model_finetune_{data}.h5",
        logging_dir=logging_dir,
    )

    test_dir = f'/Users/kcollins/invariant_face_data/illum_data/ill_{data}_test/img'
    dataset = load_dataset(test_dir, keep_file_names=True)
    acc = evaluate(model, dataset, num_samples=100, logging_dir=None)
    return acc


def complexity_heatmap(data,repr_type, all_num_ids, all_samp_per_id):

    logging_dir = f'./logging_dir/{repr_type}_{data}/'

    all_performance_metrics = []

    for num_ids in all_num_ids:
        performance_metrics = []
        for num_samp_per_id in all_samp_per_id:
            performance_metrics.append(score_setting(data,repr_type, num_ids, num_samp_per_id,logging_dir))
        all_performance_metrics.append(performance_metrics)

    plt.figure()
    sns.heatmap(all_performance_metrics)
    plt.savefig(logging_dir + "sample_complexity_heatmap.png")


if __name__ == '__main__':
    random.seed(1612)

    data = 'extreme'  # 'extreme'  # extreme or normal dataset
    repr_type = 'HOG' # vgg or hog features
    print(f'Experiment type : {data} illumination')

    all_num_ids = list(range(5,500, 150))
    all_samp_per_id = list(range(5,30,10))

    complexity_heatmap(data,repr_type, all_num_ids, all_samp_per_id)

    # num_samples_per_id = list(range(5,30,2))
    # num_ids = list(range(5,500,10))
    # explore_num_sample_per_id(data,repr_type, num_samples_per_id, num_ids=10)
    # explore_num_id(data, repr_type,num_ids=num_ids,num_samp_per_id=30)



