import numpy as np
import os
import random
from skimage import io
from skimage.feature import hog
from sklearn import decomposition
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pprint import pprint
from tensorflow.keras.models import load_model, Sequential


class TemplateModel:
    def __init__(
        self,
        template_dir,
        repr_type='HOG',
        pca_dim=-1,
        standardize=True,
        num_thresh_samples=-1,
        thresh=0.,
        num_template_ids=0,
        num_template_samples_per_id=0,
        vgg_model_path=None,
    ):
        '''
        template_dir(str): root directory of the template images
        repr_type(str): type of feature for model to use; one of 'HOG' or 'VGG' or 'RANDOM'
                        (if VGG, embeddings from the penultimate layer of pretrained VGG-Face
                        are used; if HOG, HOG features are used)
        pca_dim(int): number of principal components to project onto; if -1, do not perform PCA
        standardize(bool): whether the template representations should be standardized
        num_thresh_samples(int): number of samples to use when tuning threshold
                                 (as many as possible while remaining balanced if -1)
        thresh(float): if provided, model will use this threshold and will not tune the threshold
        num_template_ids(int): number of IDs from template dataset to use (if 0, use all)
        num_template_samples_per_id(int): number of samples per template ID to use (if 0, use all)
        '''
        self.pca = None if pca_dim == -1 else decomposition.PCA(n_components=pca_dim)
        self.standardize = standardize

        if repr_type == 'HOG':
            self.compute_feats = self.compute_hog_feats
        elif repr_type == 'VGG':
            self.compute_feats = self.compute_vgg_feats
            self.vgg_model_path = vgg_model_path
        elif repr_type == 'RANDOM':
            self.compute_feats = self.compute_random_feats
        else:
            raise ValueError("repr_type must be one of 'HOG', 'VGG', or 'RANDOM'")

        # TODO(ddoblar): add ability to customize size of dataset
        self.template_img_id_pairs = load_dataset(
            template_dir,
            num_ids=num_template_ids,
            num_samples_per_id=num_template_samples_per_id,
        )
        print('Template dataset loaded')

        # compute the projector operator
        self.template_feats = {
            int(idx): [] for idx in set(list(zip(*self.template_img_id_pairs))[1])
        }
        for img, idx in self.template_img_id_pairs:
            self.template_feats[int(idx)].append(self.compute_feats(img))
        self.template_feats = {
            idx: np.stack(feats, axis=1) for idx, feats in self.template_feats.items()
        }
        self.projector = self.compute_projector(self.template_feats)
        self.projected_templates = {
            idx: np.dot(self.projector.T, feat).T for idx, feat in self.template_feats.items()
        }
        print('Template projection operator computed')

        # tune the threshold on the template images
        self.threshold = thresh if thresh else self.tune_threshold(num_thresh_samples)

    def compute_hog_feats(self, img):
        return hog(img, block_norm='L2-Hys', transform_sqrt=True)

    def compute_vgg_feats(self, img):
        if self.vgg_model_path is None or not os.path.exists(self.vgg_model_path):
            raise FileExistsError("Please pass valid model (.hd5) file.")
        img = np.expand_dims(img, axis=0)
        model = load_model(self.vgg_model_path)
        new_model = Sequential()
        for layer in model.layers[:-1]:  # just exclude last layer from copying
            new_model.add(layer)
        activations = new_model.predict(img)
        return activations

    def compute_random_feats(self, img, dim=256):
        '''
        random features drawn from unit gaussian with dimension dim (to sanity check rest of model)
        '''
        return np.random.normal(size=256)

    def compute_projector(self, template_feats):
        '''
        compute operator to project onto feature space
        '''
        all_template_feats = np.hstack(template_feats.values())
        if self.standardize:
            # TODO(ddoblar): figure out whether the transposes should be here or not
            all_template_feats = StandardScaler().fit_transform(all_template_feats)
            # all_template_feats = StandardScaler().fit_transform(all_template_feats.T).T
        if self.pca:
            all_template_feats = self.pca.fit_transform(all_template_feats)
        return all_template_feats

    def score(self, ex1, ex2):
        '''
        compute similarity score for ex1 and ex2
        '''
        ex1_feats = cosine_similarity(self.projector.T, self.compute_feats(ex1).reshape(1, -1)).T
        ex2_feats = cosine_similarity(self.projector.T, self.compute_feats(ex2).reshape(1, -1)).T

        ex1_cosine_sims = {
            idx: cosine_similarity(feats, ex1_feats)
            for idx, feats in self.projected_templates.items()
        }
        ex2_cosine_sims = {
            idx: cosine_similarity(feats, ex2_feats)
            for idx, feats in self.projected_templates.items()
        }

        # perform mean pooling
        ex1_mean_pool = np.array([np.mean(sim) for sim in ex1_cosine_sims.values()]).reshape(1, -1)
        ex2_mean_pool = np.array([np.mean(sim) for sim in ex2_cosine_sims.values()]).reshape(1, -1)

        # compute score as normalized dot product
        score = cosine_similarity(ex1_mean_pool, ex2_mean_pool).item()
        return score

    def predict(self, ex1, ex2):
        '''
        predict whether ex1 and ex2 are same identity
        '''
        score = self.score(ex1, ex2)
        return int(score > self.threshold)

    def evaluate(self, dataset, num_samples=-1):
        '''
        Evaluates the model on pairs of examples in the dataset specified by dataset_path.
        Returns accuracy on sampled pairs of examples in the dataset. Classes are balanced such
        that there are equal number of same and different pairs in the evaluation set.

        num_samples(int): num_samples//2 pairs will be sampled with (1) the same label and
                          (2) different labels for tuning the threshold. If -1, use as many
                          samples as possible while keeping classes balanced.
        '''
        print('Evaluating model:')
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

        num_correct = 0
        num_pairs = 0
        wrong_pairs = []

        for idx1, idx2 in tqdm(pairs):
            ex1, label1 = dataset[idx1]
            ex2, label2 = dataset[idx2]
            output = self.predict(ex1, ex2)
            label = int(label1 == label2)
            num_pairs += 1
            if output == label:
                num_correct += 1
            else:
                wrong_pairs.append((label1, label2))

        accuracy = num_correct / num_pairs
        # print(f"[evaluate] num_correct : {num_correct}")
        # print(f"[evaluate] num_pairs : {num_pairs}")
        print(f"[evaluate] accuracy : {accuracy}")
        # print(f"wrong_pairs :")
        # pprint(wrong_pairs)
        # print(f'number wrong with same label : {sum([a == b for a, b in wrong_pairs])}')
        # print(f'number wrong with diff label : {sum([a != b for a, b in wrong_pairs])}')
        return accuracy

    def tune_threshold(self, num_samples=-1):
        '''
        Tune threshold using the template examples.

        num_samples pairs are sampled from all samples in the template set such that there are
        equal number same ID and different ID pairs. Threshold is tuned by computing the accuracy
        for each bin of possible thresholds (a threshold between two scores in the sampled pair
        in sorded order will not change the accuracy), then choosing the threshold to be in the
        center of the bin of possible thresholds that maximizes accuracy on the sampled template
        pairs.

        num_samples(int): num_samples//2 pairs will be sampled with (1) the same label and
                          (2) different labels for tuning the threshold. If -1, use as many
                          samples as possible while keeping classes balanced.

        Returns: tuned threshold(float)
        '''
        print('Tuning threshold:')
        dataset = self.template_img_id_pairs
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

        score_label_pairs = []
        for idx1, idx2 in tqdm(pairs):
            ex1, label1 = dataset[idx1]
            ex2, label2 = dataset[idx2]
            score = self.score(ex1, ex2)
            label = int(label1 == label2)
            score_label_pairs.append((score, label))
        score_label_pairs.sort(key=lambda x: x[0])

        # choose the thresh that minimizes the number of misclassified pairs
        template_scores = list(zip(*score_label_pairs))[0]
        # this is not as efficient as it could be, but it's plenty fast
        accuracies = [scores_to_acc(score_label_pairs, thresh) for thresh in template_scores]
        best_idx = max(enumerate(accuracies), key=lambda x: x[1])[0]
        # take the average of the scores on the inflection point as the threshold
        thresh = (template_scores[best_idx] + template_scores[best_idx+1]) / 2
        print(f'Tuned threshold : {thresh}')

        template_acc = scores_to_acc(score_label_pairs, thresh)
        print(f"Accuracy on the templates with tuned threshold : {template_acc}")

        return thresh


def scores_to_acc(score_label_pairs, thresh):
    return sum([int(int(score > thresh) == label) for score, label in score_label_pairs]) /\
        len(score_label_pairs)


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


if __name__ == '__main__':
    random.seed(1612)

    data = 'normal'  # 'extreme'  # extreme or normal dataset
    print(f'Experiment type : {data} illumination')

    template_dir = f'/Users/dylan/projects/face-illumination-invariance/data/ill_{data}_debug/img'
    model = TemplateModel(
        template_dir,
        repr_type='HOG',
        # repr_type='RANDOM',  # sanity check that with large sample sizes accuracy is nearly 0.5
        pca_dim=100,
        standardize=True,
        num_thresh_samples=100,
        # thresh=0.9999560161509551,  # pass in a threshold if you don't want to tune
        num_template_ids=10,
        num_template_samples_per_id=30,
    )

    test_dir = f'/Users/dylan/projects/face-illumination-invariance/data/ill_{data}_overfit/img'
    # dataset = load_dataset(template_dir)  # sanity check: model does well on the template set
    dataset = load_dataset(test_dir)
    acc = model.evaluate(dataset, num_samples=100)
    print(f"model accuracy on the balanced test set : {acc}")
