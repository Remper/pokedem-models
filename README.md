# pokedem-models [![CI status][ci-img]][ci]

Training and model generation code for pokedem-plus

## Examples

### API startup

```api.py``` implements an entry point for the API that exposes a single trained model via ```/predict?features=[x1,x2,x3,...]``` method

```
python api.py --input model
```

Where ```model``` is produced by training scripts described below


### Basic deep neural network

```train.py``` implements multiple dense layers with tanh as an activation
function with a softmax on top. Training set should be in SVM format.

```
python train.py --train features.svm
```

## Installation

1. Install following ubuntu packets (or use their counterparts in other systems/build from sources):
    * libblas-dev
    * liblapack-dev
    * gfortran
    * python-numpy
    * python-scipy
    * python3-numpy
    * python3-scipy
    
2. Install python dependencies:

    ```pip install -r requirements.txt```
    
3. Install Tensorflow for your system using the official [guide](https://www.tensorflow.org/versions/r0.9/get_started/index.html)

[ci-img]: https://travis-ci.com/Remper/pokedem-models.svg?branch=master&token=QTsnxbPSaywz8CsQ1xCH
[ci]:     https://travis-ci.com/Remper/pokedem-models