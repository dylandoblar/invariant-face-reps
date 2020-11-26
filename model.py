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
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation, Dense
import tensorflow as tf


from utils import *


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
        threshold_data_path=None,
        logging_dir=None,
    ):
        '''
        Implementation of the model described in Figure 5 of "Learning invariant representations
        and applications to face verification" by Liao et al., with the ability to use features
        from VGG-Face instead of HOG features in the feature extraction phase (the penultimate
        activations of a pre-trained VGG-Face model are used in this case). Refer to the paper
        for model details:
        https://papers.nips.cc/paper/2013/file/ad3019b856147c17e82a5bead782d2a8-Paper.pdf

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
        vgg_model_path(str): if using VGG model, provide path to the model .h5 file
        logging_dir(str): where to save data, if desired
        '''
        self.pca = None if pca_dim == -1 else decomposition.PCA(n_components=pca_dim)
        self.standardize = standardize

        if repr_type == 'HOG':
            self.compute_feats = self.compute_hog_feats
        elif repr_type == 'VGG':

            tf.autograph.set_verbosity(0)
            if vgg_model_path is None or not os.path.exists(vgg_model_path):
                raise FileExistsError("Please pass valid model (.h5) file.")
            self.vgg_model_path = vgg_model_path
            # model = create_vggface_model(self.vgg_model_path, 800)
            model = load_model(self.vgg_model_path)
            new_model = Sequential()
            for layer in model.layers[:-1]:  # just exclude last layer from copying
                new_model.add(layer)
            self.model = new_model
            self.compute_feats = self.compute_vgg_feats
            self.vgg_model_path = vgg_model_path
        elif repr_type == 'RANDOM':
            self.compute_feats = self.compute_random_feats
        else:
            raise ValueError("repr_type must be one of 'HOG', 'VGG', or 'RANDOM'")

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
        self.threshold = thresh if thresh else self.tune_threshold(num_thresh_samples,logging_dir)

    def compute_hog_feats(self, img):
        return hog(img, block_norm='L2-Hys', transform_sqrt=True)

    def compute_vgg_feats(self, img):
        img = np.expand_dims(img, axis=0)
        activations = self.model.predict(img)[0]
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
        Compute the similarity score for ex1 and ex2.

        ex1(ndarray): array of pixels for the first identity
        ex2(ndarray): array of pixels for the second identity
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
        Predict whether ex1 and ex2 have the same identity.

        ex1(ndarray): array of pixels for the first identity
        ex2(ndarray): array of pixels for the second identity
        '''
        score = self.score(ex1, ex2)
        return int(score > self.threshold)

    def tune_threshold(self, num_samples=-1,logging_dir=None):
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
        logging_dir(str): where to save threshold data and/or plots

        Returns: tuned threshold(float)
        '''
        print('Tuning threshold:')
        pairs = gen_balanced_pairs_from_dataset(self.template_img_id_pairs, num_samples)

        score_label_pairs = []
        for idx1, idx2 in tqdm(pairs):
            ex1, label1 = self.template_img_id_pairs[idx1]
            ex2, label2 = self.template_img_id_pairs[idx2]
            score = self.score(ex1, ex2)
            label = int(label1 == label2)
            score_label_pairs.append((score, label))
        score_label_pairs.sort(key=lambda x: x[0])

        # choose the thresh that minimizes the number of misclassified pairs
        template_scores = list(zip(*score_label_pairs))[0]
        # this is not as efficient as it could be, but it's plenty fast
        accuracies = [scores_to_acc(score_label_pairs, thresh) for thresh in template_scores]
        if logging_dir is not None:
            if not os.path.exists(logging_dir): os.makedirs(logging_dir)
            write_csv(score_label_pairs, ["score","label"],logging_dir+"threshold_data.csv")
            plot_roc(score_label_pairs,logging_dir+"threshold_roc.png")

        best_idx = max(enumerate(accuracies), key=lambda x: x[1])[0]
        # take the average of the scores on the inflection point as the threshold
        thresh = (template_scores[best_idx] + template_scores[best_idx+1]) / 2
        print(f'Tuned threshold : {thresh}')

        template_acc = scores_to_acc(score_label_pairs, thresh)
        print(f"Accuracy on the templates with tuned threshold : {template_acc}")


        return thresh

def create_vggface_model(model_weights, num_identities=100):
    # help from: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

    # same architecture as VGGFace
    vggface_model = Sequential()
    vggface_model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    vggface_model.add(Convolution2D(64, (3, 3), activation='relu'))
    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(64, (3, 3), activation='relu'))
    vggface_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(128, (3, 3), activation='relu'))
    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(128, (3, 3), activation='relu'))
    vggface_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(256, (3, 3), activation='relu'))
    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(256, (3, 3), activation='relu'))
    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(256, (3, 3), activation='relu'))
    vggface_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vggface_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vggface_model.add(ZeroPadding2D((1, 1)))
    vggface_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vggface_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vggface_model.add(Convolution2D(4096, (7, 7), activation='relu'))
    vggface_model.add(Dropout(0.5))
    vggface_model.add(Convolution2D(4096, (1, 1), activation='relu'))
    vggface_model.add(Dropout(0.5))
    vggface_model.add(Convolution2D(2622, (1, 1)))
    # vggface_model.add(Convolution2D(2622, (1, 1)))
    vggface_model.add(Flatten(name="activations"))
    vggface_model.add(Activation('softmax'))

    model = Sequential()
    for layer in vggface_model.layers[:-1]:model.add(layer)
    # freeze the weights to allow only pretraining on final layer and prior layer from original model
    for layer in model.layers[:-3]:layer.trainable = False
    model.add(Dense(num_identities, activation='softmax', name='predictions'))

    # compile final model
    # retrain (but just the last layer since all else frozen)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights(model_weights)

    return model