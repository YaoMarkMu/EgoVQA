## Ego-VQA:  Egocentric Video Question Answering dataset

### Ego-VQA Dataset
In this experiment, we use [Ego-VQA dataset](https://github.com/fanchenyou/EgoVQA/tree/master/data).
Please cite this paper[[27]](http://homes.sice.indiana.edu/fan6/docs/EgoVQA.pdf) if you use this dataset.
The original IU Multi-view egocentric video dataset can be downloaded [[here]](http://vision.soic.indiana.edu/identifying-1st-3rd/).
Though it's not necessary to download the original frames, it would be helpful to visualize the egocentric data.

### Third-person VQA v.s. Ego-VQA
![Task](/pics/video_sample.png)  



### Pre-training Dataset (YouTube2Text-QA)
Because the Ego-VQA is a generally small QA dataset, we use a large YouTube2Text-QA dataset for pre-training.
Please cite this paper[[27]](https://arxiv.org/abs/1707.06355) if you use this dataset.



### Pre-train
~~~~
python pretrain.py --memory_type=_mrm2s | _stvqa | _enc_dec | _co_mem
~~~~
Please manually set select the best model on validation set for each model (memory_type), 
and modify [train.py](https://github.com/fanchenyou/EgoVQA/blob/master/train.py#L149) accordingly to update the initialization models.

### Train, validate, and test
For training and validating, execute the following command
~~~~
python train.py --split=1|2|3 --memory_type=_mrm2s | _stvqa | _enc_dec | _co_mem
~~~~

For testing, execute the following command
~~~~
python test.py --memory_type=_mrm2s | _stvqa | _enc_dec | _co_mem
~~~~

Please modify train.py to manually add your pre-trained models to initialize the entire model.
Current we use default models we provided in previous steps.


### Requirements
Python = 2.7
 
PyTorch = 1.0+ [[here]](https://pytorch.org/)

GPU training with 4G+ memory, testing with 1G+ memory.

