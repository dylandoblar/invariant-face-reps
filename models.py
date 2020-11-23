import numpy as np
import os
from skimage.feature import hog
from skimage import io
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.metrics.pairwise import cosine_similarity


class TemplateModel:
    def __init__(self, template_dir, repr_type='HOG', pca_dim=0, standardize=True, thresh=0.99999):
        '''
        pca_dim(int): if 0, do not perform PCA
        '''
        self.pca = None if pca_dim == 0 else decomposition.PCA(n_components=pca_dim)
        self.standardize = standardize
        self.threshold = thresh
        if repr_type == 'HOG':
            self.compute_feats = self.compute_hog_feats
        elif repr_type == 'VGG':
            self.compute_feats = self.compute_vgg_feats
        else:
            raise ValueError("repr_type must be one of 'HOG' or 'VGG'")
        # TODO(ddoblar): add ability to customize size of dataset
        self.template_img_id_pairs = load_dataset(template_dir)
        print('template dataset loaded')
        self.template_feats = {
            int(idx): [] for idx in set(list(zip(*self.template_img_id_pairs))[1])
        }
        for img, idx in self.template_img_id_pairs:
            self.template_feats[int(idx)].append(self.compute_feats(img))
        self.template_feats = {
            idx: np.stack(feats, axis=1) for idx, feats in self.template_feats.items()
        }
        self.projector = self.compute_projector(self.template_feats)
        # print(f'self.projector.shape: {self.projector.shape}')
        self.projected_templates = {
            idx: np.dot(self.projector.T, feat).T for idx, feat in self.template_feats.items()
        }

    def compute_hog_feats(self, img):
        # TODO(ddoblar): verify that default params are good
        return hog(img, block_norm='L2-Hys')

    def compute_vgg_feats(self, img):
        raise NotImplementedError

    def compute_projector(self, template_feats):
        '''
        compute operator to project onto feature space
        '''
        all_template_feats = np.hstack(template_feats.values())
        # print(f'all_template_feats.shape: {all_template_feats.shape}')
        if self.standardize:
            all_template_feats = StandardScaler().fit_transform(all_template_feats)  # transpose?
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
        # print(f"ex1_mean_pool : {ex1_mean_pool}")
        # print(f"ex2_mean_pool : {ex2_mean_pool}")

        # compute score as normalized dot product
        score = cosine_similarity(ex1_mean_pool, ex2_mean_pool).squeeze()
        print(f"score : {score}")
        return score

    def predict(self, ex1, ex2):
        '''
        predict whether ex1 and ex2 are same identity
        '''
        score = self.score(ex1, ex2)
        return int(score > self.threshold)

    def tune_threshold(self):
        '''
        tune threshold using the template examples
        '''
        # TODO(ddoblar): minimize error on templates
        raise NotImplementedError


def load_dataset(dataset_dir):
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
    and returns a generator of (X, y) tuples where X is an array of pixels and y is the ID
    '''
    labels = os.listdir(dataset_dir)
    example_fname_label_pairs = [
        (os.path.join(dataset_dir, label, fname), label)
        for label in labels
        for fname in os.listdir(os.path.join(dataset_dir, label))
    ]
    img_label_pairs = [(io.imread(fname), label) for fname, label in example_fname_label_pairs]
    return img_label_pairs


if __name__ == '__main__':
    template_dir = '/Users/dylan/projects/face-illumination-invariance/data/ill_normal_debug/img'
    model = TemplateModel(template_dir, repr_type='HOG', pca_dim=100, standardize=True)
    test_dir = '/Users/dylan/projects/face-illumination-invariance/data/ill_normal_overfit/img'
    dataset = load_dataset(test_dir)
    print(model.predict(dataset[-45][0], dataset[-1][0]))  # should be smaller
    print(model.predict(dataset[-145][0], dataset[-1][0]))  # should be smaller
    print(model.predict(dataset[-1][0], dataset[-1][0]))  # should be = 1 since same image
    print(model.predict(dataset[-2][0], dataset[-1][0]))  # should be close to 1 since same ID
