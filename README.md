# STFGNN-PaddlePaddle
A PaddlePaddle re-implementation of the STFGNN model described in https://arxiv.org/abs/2012.09641.  

I have also written some [notes](STFGNN.pdf) explaining the model.


## Contents
`main.py` contains the training loop    

`models/model.py` contains the model    

`utils.py` contains the generation of training/validation/test set and the preprocessing procedures.   

`temporal_graph_gen.py` contains the generation of the DTW matrix.  
For details of the Dynamic Time Warping(DTW) algorithm, please refer to [wiki](https://en.wikipedia.org/wiki/Dynamic_time_warping) or the [paper](https://arxiv.org/abs/2012.09641).

`config/individual_3layer_12T.json` contains the hyperparameter configuration for the model.

## Requirements and Training 
`sh run.sh` to install all the requirements and run a demo of the program to see the results.  
You should observe the loss decreasing and the accuracy over 80%
  
if you have already downloaded all the requirements, you may use  
`python main.py --config config/individual_3layer_12T.json` to start training. 
