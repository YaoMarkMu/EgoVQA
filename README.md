### Experiment on Ego-VQA

### Ego-VQA Dataset
In this experiment, we use [[Ego-VQA dataset]]().
Please cite this paper[[27]](http://homes.sice.indiana.edu/fan6/docs/EgoVQA.pdf) if you use this dataset.
The original IU Multi-view egocentric video dataset can be downloaded [[here]](http://vision.soic.indiana.edu/identifying-1st-3rd/).
Though it's not necessary to download the original frames, it would be helpful to visualize the egocentric data.


### Pre-training Dataset (YouTube2Text-QA)
Because the Ego-VQA is a generally small QA dataset, we use a large [[YouTube2Text-QA dataset]] for pre-training.
Please cite this paper[[27]](http://homes.sice.indiana.edu/fan6/docs/EgoVQA.pdf) if you use this dataset.
The original IU Multi-view egocentric video dataset can be downloaded [[here]](http://vision.soic.indiana.edu/identifying-1st-3rd/).
Though it's not necessary to download the original frames, it would be helpful to visualize the egocentric data.



### Train, validate, and test
For training and validating, execute the following command
~~~~
python main.py
~~~~

For testing, just add a --test=1 flag, such as
~~~~
python main.py --test=1
~~~~

Please modify train.py to test your own models. 
Current we use default models we provided in previous steps.


