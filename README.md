# Face verification with illumination-invariant representations

We investigate invariance properties of face representations under varying illumination conditions.  From scratch, we implement and extend the model proposed by [1] which uses a templates-and-signatures approach to generate unique representations for query points unseen during training.  The model extracts face features from a set of template images, where several face identities have a small number of example images under different transformations. In our case, we consider transformations involving both natural and extreme illumination conditions; the original work only considers 2D affine transformations as well as yaw rotations. Further, the raw features originally used in [1] are histograms of oriented gradients (HOG); we evaluate the performance of such a feature extractor against an equivalent model that uses the penultimate activations of a pre-trained VGG-Face network [2] (a VGG16-based CNN designed and trained for facial recognition) as the raw feature extractor. After the feature extraction step, the model — which is formulated identically for both the HOG and VGG-Face features — then computes the cosine similarity between all template identities and each image in a query pair to obtain the signature per image.  If the signatures of the query images are close (e.g., have cosine similarity above some threshold), the model predicts that they are the same identity.  We perform experiments to determine whether this templates-and-signatures model can learn illumination-invariant face representations, and whether the deep convolutional architecture of VGG-Face produces features which are more effective than HOG representations.  

We discover that under natural illumination settings, both model types effectively solve the face verification task. Additionally, HOG and VGG-Face are impressively invariant to even extreme variations in illumination. On synthetic data, HOG largely outperforms; however, only VGG-Face extends to real-world images. Interestingly, we also find that natural illumination templates yields better generalization for identification under novel extreme illumination, possibly because of the more accessible structural information in naturally lit images. 

Future work could investigate this hypothesis and extend the training regime to include out-of-distribution samples (i.e., through domain randomization). Additional next steps include exploring more complicated, combinatorial transformations, such as pose with illumination, with which humans excel, as well as comparing the HOG and deep-net based models against a graphics-engine based approach. Cognitive plausibility of these models could be assessed against human experiments run on Amazon Mechanical Turk.

## Running the code

Install the necessary dependenceies by running:
```
pip install -r requirements.txt
```

The template model is implemented in `model.py`, and can be initialized with either HOG or VGG-Face features.  The evaluation pipeline for the model is implemented in `eval.py`, along with several functions useful in running our experiments.  The dataloader and several useful evaluation and plotting functions are in `utils.py`.  To run our full experimentation and plotting pipeline, run 
```
python eval.py
```

## References

[1] Liao, Qianli, Joel Z. Leibo, and Tomaso Poggio. "Learning invariant representations and applications to face verification." _Advances in neural information processing systems_. 2013.

[2] Parkhi, Omkar M., Andrea Vedaldi, and Andrew Zisserman. "Deep face recognition." 2015.
