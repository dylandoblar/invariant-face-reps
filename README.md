# Biologically plausible illumination-invariant face representations
Dylan D. Doblar (ddoblar@mit.edu), Katherine M. Collins, Bernhard Egger, Tomaso Poggio; Massachusetts Institute of Technology

Humans possess a remarkable ability to identify objects — and faces in particular — under highly variable conditions. The invariance hypothesis (Leibo et al., 2015) posits that the goal of the human ventral visual system is to learn representations that are invariant to identity-preserving transformations. One computational approach to learn such representations is the templates-and-signatures model proposed by Liao et al. (2013). The model measures similarities between template images and unseen query images to produce discriminative and invariant signatures. From a small number of template examples, the model can learn invariance to group transformations (e.g., scaling, translation, and in-plane rotation) for new object classes, whereas object-class-specific templates are required for non-group transformations (e.g., pose). Here, we probe the capacity of this approach to handle variation in illumination — a complex set of non-group transformations — on a face verification task. A 3D morphable face model is used to generate synthetic datasets under both natural and extreme illumination. We benchmark the templates-and-signatures approach against VGG16, a convolutional neural network (CNN), and compare the effects of a generic object-class versus domain-specific learned prior by pre-training VGG16 either on ImageNet or human faces. We find that under natural illumination settings, the templates-and-signatures model effectively solves the face verification task, outperforming both CNN variants. Additionally, the templates-and-signatures model’s learned representations are impressively invariant to extreme variations in illumination and generalize best when using natural illumination templates. These invariances hold even with tens of training examples, which is particularly striking behavior relative to the CNNs that have been pre-trained on millions of images. Coupled with its simplicity and its implications for a biologically plausible sequence of class-specific developmental periods for learning invariances, the model’s ability to generalize to out-of-distribution illumination settings from few examples lends credence to a templates-and-signatures account of feed-forward object recognition.

Acknowledgements: We thank Qianli Liao for his advice on how to implement the templates-and-signatures model. This work was funded in part by the Center for Brains, Minds and Machines (CBMM), NSF STC award CCF-1231216. B. Egger is supported by a PostDoc Mobility Grant, Swiss National Science Foundation P400P2 191110.


## Running the code

Install the necessary dependenceies by running:
```
pip install -r requirements.txt
```

The template model is implemented as `TemplateModel` in `model.py`, and can be initialized with either HOG or VGG-Face features.  The evaluation pipeline for the model is implemented in `eval.py`, along with several functions useful in running our experiments.  The dataloader and several useful evaluation and plotting functions are in `utils.py`.  To run our full experimentation and plotting pipeline, run 
```
python eval.py
```

## V-VSS 2021 Presentation
```
TODO(ddoblar): upload video somewhere and link it here
```

## Citation
If you use this code in your work please cite our V-VSS abstact:
```
TODO(ddoblar): bibtex citation
```
