# STFGNN-PaddlePaddle
A PaddlePaddle re-implementation of the STFGNN model described in https://arxiv.org/abs/2012.09641.  

I have also written some [notes](STFGNN.pdf) explaining the model.


## Contents
`main.py` contains the training loop  
`models/model.py` contains the model  
`utils.py` contains the generation of training/validation/test set and the preprocessing procedures.  
`temporal_graph_gen.py` contains the generation of the dtw matrix. For details of the DTW algorithm, please refer to [wiki](https://en.wikipedia.org/wiki/Dynamic_time_warping) or the [paper](https://arxiv.org/abs/2012.09641)

## Requirements and Training 
`sh requirements.sh` to install all the requirements   
`sh run.sh` to run a demo of the program and see the results.
