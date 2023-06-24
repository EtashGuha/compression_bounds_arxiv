# Ground Zero
## Quick and extendable experimentation with classification models

### Setup
```
conda update -n base -c defaults conda
conda create -n groundzero python==3.10
conda activate groundzero
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
python -m pip install -e .
```

### Workflow
Base code goes in `groundzero`. Experiment code goes in `exps`. Config files go in `cfgs`.

### To Run Experiments
To run the experiments for CIFAR10, run 
```
python groundzero/main.py -c cfgs/cifar10.yaml --model mlp
```
To run the experiments for MNIST, run 
```
python groundzero/main.py -c cfgs/mnist.yaml --model mlp
```
A pickle file in results_dir will be created with name "results_dir/mnist.pkl" for MNIST and "results_dir/cifar10.pkl" for CIFAR10. In pickle file, there will be our generalization bounds, the true generalization errors, the margins, the $d$ value used for our compression, the bounds from Neyshabur 2015, the bounds from Bartlett 2017, the bounds from Neyshabur 2017, and the list of epochs. 

