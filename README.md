# Refinement for Point Cloud Completion
Pytorch code for the paper "Refinement of Predicted Missing Parts Enhance Point Cloud Completion".

![](figs/principal.png)

Given a partial input, our method predicts and refine the missing part to obtain an integrated completed point cloud.

## Setup
We use Ubuntu 18.04 with Python 3.6.9. and CUDA 10.2. Follow the next instructions to establish the environment.

1. Create an environment
~~~
python -m venv Refinement --system-site-packages
source Refinement/bin/activate
~~~

2. Install dependencies
~~~
pip install torch==1.3.1 torchvision
pip install -r requirements.txt
~~~

## Training
### 1. Download the ShapeNet dataset
~~~
cd data
bash get_dataset.sh
~~~

### 2. Compile the extension modules
~~~
cd extensions/chamfer_dist
python setup.py install

cd losses/emd
python setup.py install
~~~

### 3. Train a network
We use Visdom to see the training progress. Therefore, before training, we need to start a visdom server with the command:

~~~
visdom -port 8997
~~~

Port 8997 is used by default by our training scripts. If you want to use a different port, be careful to change the parameter in the training scripts.

To train our model, use the command:
~~~
python train_MBD.py --model=MBD_SHAPENET
~~~

The trained model and training log will be stored in the folder ./log/MBD_SHAPENET. Please check the Python code to set other parameters such as batch size or number of epochs. Parameters by default were used to get the paper's results.

We also provide the Python code to train our MLP variant (train_MLP.py), the fully convolutional auto_encoder (train_FCAE.py) and the MSN method (train_MSN.py). Just note that MSN method requires Pytorch 1.2 and the compilation of losses/chamfer, losses/expansion_penalty, and losses/MDS.

### 4. Evaluation
Download the pretrained models from [here](https://drive.google.com/drive/folders/14i51epgfmftfBY4R569XZl6Ia9GLWJA_?usp=sharing). To evaluate the methods and get the results reported in the paper, you must proceed as follows:

1. Perform the evaluation of our model first using the script *evaluate_model.sh*. The evaluation of our model creates a set random input shapes and store them in a folder. This test set must be used by the other evaluation scripts to make a fair comparison. 

2. Perform the evaluation of compared methods using the command:
    ~~~
    python shapenet_model_evaluation_FCAE.py --model=pretrained/FCAE_SHAPENET/ --inputTestFolder=Results/
    ~~~

Optionally, you can also save the results of the compared methods by setting the option --outputFolder in the evaluation scripts.