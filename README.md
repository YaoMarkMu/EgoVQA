## Ego-VQA:  Egocentric Video Question Answering dataset

### Ego-VQA Dataset
In this experiment, we use [Ego-VQA dataset](https://github.com/fanchenyou/EgoVQA/tree/master/data).
Please cite this paper[[27]](http://homes.sice.indiana.edu/fan6/docs/EgoVQA.pdf) if you use this dataset.
The original IU Multi-view egocentric video dataset can be downloaded [[here]](http://vision.soic.indiana.edu/identifying-1st-3rd/).
Though it's not necessary to download the original frames, it would be helpful to visualize the egocentric data.


### Pre-training Dataset (YouTube2Text-QA)
Because the Ego-VQA is a generally small QA dataset, we use a large YouTube2Text-QA dataset for pre-training.
Please cite this paper[[27]](https://arxiv.org/abs/1707.06355) if you use this dataset.



### Pre-train
~~~~
python pretrain.py ----memory_type=_mrm2s | _stvqa | _enc_dec | _co_mem
~~~~

### Train, validate, and test
For training and validating, execute the following command
~~~~
python train.py --split=1|2|3 ----memory_type=_mrm2s | _stvqa | _enc_dec | _co_mem
~~~~

For testing, just add a --test=1 flag, such as
~~~~
python test.py ----memory_type=
~~~~

Please modify train.py to manually add your pre-trained models to initialize the entire model.
Current we use default models we provided in previous steps.


