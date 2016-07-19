# pokedem-models 

![](https://travis-ci.com/Remper/pokedem-models.svg?branch=master&token=QTsnxbPSaywz8CsQ1xCH)

Training and model generation code for pokedem-plus

## Examples

### Basic deep neural network

```train.py``` implements multiple dense layers with tanh as an activation
function with a softmax on top. Training set should be in SVM format.

```
python train.py --train features.svm
```