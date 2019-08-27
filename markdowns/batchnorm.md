[Introduction](https://github.com/vinsis/understanding-neuralnetworks-pytorch/blob/master/batchnorm.md#introduction)

[An intuitive example of internal covariate shift](https://github.com/vinsis/understanding-neuralnetworks-pytorch/blob/master/batchnorm.md#an-intuitive-example-of-internal-covariate-shift)

[Solution to internal covariate shift](https://github.com/vinsis/understanding-neuralnetworks-pytorch/blob/master/batchnorm.md#solution-to-internal-covariate-shift)

[Batch Normalization may not always be optimal for learning](https://github.com/vinsis/understanding-neuralnetworks-pytorch/blob/master/batchnorm.md#batch-normalization-may-not-always-be-optimal-for-learning)

[Batch Normalization with backpropagation](https://github.com/vinsis/understanding-neuralnetworks-pytorch/blob/master/batchnorm.md#batch-normalization-with-backpropagation)

[Batch Normalization for 2D data](https://github.com/vinsis/understanding-neuralnetworks-pytorch/blob/master/batchnorm.md#batch-normalization-for-2d-data)

[Batch Normalization for 3D inputs](https://github.com/vinsis/understanding-neuralnetworks-pytorch/blob/master/batchnorm.md#batch-normalization-for-3d-inputs)

[Batch Normalization for images (or any 4D input)](https://github.com/vinsis/understanding-neuralnetworks-pytorch/blob/master/batchnorm.md#batch-normalization-for-images-or-any-4d-input)

### Introduction

Let's start with a discussion of what problem were the authors of the [original paper](https://arxiv.org/abs/1502.03167) dealing with when they came up with the idea of Batch Normalization:

> [...] the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs.

### An intuitive example of internal covariate shift
Let's first talk about covariate shift. A covariate shift occurs when:

i) given the same observation X = x, the conditional distributions of Y in training and test sets are the same BUT
<br>
ii) marginal distribution of X in training set is different from marginal distribution of X in test set.

In other words:
<br>P<sub>train</sub>[Y|X=x] = P<sub>test</sub>[Y|X=x] BUT <br>P<sub>train</sub>[X] ≠ P<sub>test</sub>[X]

Now let's talk about the second layer _l_ inside a network with just two layers. The layer takes in an input and spits out an output. The output depends on the input (X) as well as the parameters of the network (θ). It can be thought of as a single layer neural network trying to learn P[Y|X,θ] where θ is the set of network parameters. Now if the distribution of θ changes during the training process, the network has to re-learn P[Y|X,θ<sub>new</sub>]. This is the internal covariate shift problem. This may not be easy or fast if the network finds itself spending more time in linear extremities of non-linear functions like sigmoid or tanh. Quoting from the paper:

> Consider a network computing<br>
`l = F2(F1(u, Θ1), Θ2)`<br>
where F1 and F2 are arbitrary transformations, and the parameters Θ1,Θ2 are to be learned so as to minimize the loss l. Learning Θ2 can be viewed as if the inputs x = F1 (u, Θ1 ) are fed into the sub-network
l = F2(x,Θ2). For example, a gradient descent step<br>
Θ2 <- Θ2 - (α/m)\*Σ<sub>m</sub>(∂F2(xi,Θ2)/∂Θ2)<br>
(for batch size m and learning rate α) is exactly equivalent to that for a stand-alone network F2 with input x. There- fore, the input distribution properties that make training more efficient – such as having the same distribution be- tween the training and test data – apply to training the sub-network as well. As such it is advantageous for the distribution of x to remain fixed over time. Then, Θ2 does not have to readjust to compensate for the change in the distribution of x.

So the solution is to normalize the inputs.


### Solution to internal covariate shift

If a neuron is able to see the entire range of values across the entire training set that it is going to get as subsequent inputs, training can be made faster by normalizing the distribution (i.e. making its mean zero and variance one). The authors call it _input whitening_:

> It has been long known (LeCun et al., 1998b; Wiesler & Ney, 2011) that the network training converges faster if its inputs are whitened – i.e., linearly transformed to have zero means and unit variances, and decorrelated. As each layer observes the inputs produced by the layers below, it would be advantageous to achieve the same whitening of the inputs of each layer. By whitening the inputs to each layer, we would take a step towards achieving the fixed distributions of inputs that would remove the ill effects of the internal covariate shift.

However, for a large training set, it is not computationally feasible to normalize inputs for each neuron. Hence, the normalization is carried out on a per-batch basis:

> In the batch setting where each training step is based on the entire training set, we would use the whole set to normalize activations. However, this is impractical when using stochastic optimization. Therefore, we make the second simplification: since we use mini-batches in stochastic gradient training, each mini-batch produces estimates of the mean and variance of each activation. This way, the statistics used for normalization can fully participate in the gradient backpropagation.

### Batch Normalization may not always be optimal for learning

Batch normalization may lead to, say, inputs always lying in the linear range of sigmoid function. Hence the inputs are shifted and scaled by parameters γ and β. Optimal values of these hyperparameters can be __learnt__ by using backpropagation.

> Note that simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity. To address this, we make sure that the transformation inserted in the network can represent the identity transform. To accomplish this, we introduce, for each activation x(k) , a pair of parameters γ(k), β(k), which scale and shift the normalized value:<br>
y(k) = γ(k)x<sub>normalized</sub>(k) + β(k).

### Batch Normalization with backpropagation

If backpropagation ignores batch normalization, it may undo the modification due to batch normalization. Hence, backprop needs to incorporate batch normalization while calculating gradients. [This link](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html) provides a nice explanation of how to derive backpropagation gradients for batch normalization. Let's see it in action now:

### Batch Normalization for 2D data

Let's start with 1D normalization.
```python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
```

Let's define X and calculate its mean and variance. Note that X has size (20,100) which means it has 20 samples with each sample having 100 features (or dimensions). For normalizing, we need to look across all samples. In other words, we normalize across 20 values for each dimension (with each value coming on a sample).

```python
X = torch.randn(20,100) * 5 + 10
X = Variable(X)

mu = torch.mean(X[:,1])  
var_ = torch.var(X[:,1], unbiased=False)
sigma = torch.sqrt(var_ + 1e-5)
x = (X[:,1] - mu)/sigma
```

Note that in the line above we set `unbiased = False` while calculating `var_`. This is to prevent [Bessel's correction](https://en.wikipedia.org/wiki/Bessel's_correction). In other words, we want to divide by N and not N-1 while calculating the variance. `x` is the same as the result of batch normalization. Also note that in the code below we set `affine = False`. This is to prevent creation of parameters γ(k), β(k) which may scale and shift normalized data again. While training, `affine` is set to `True`.

```python
B = nn.BatchNorm1d(100, affine=False)
y = B(X)
print(x.data / y[:,1].data)
```

Output:
```
 1.0000
 1.0000
 1.0000
 ...
 ```
 This also works for 3D input.

 ### Batch Normalization for 3D inputs

 Let's define some 3D data with mean 4 and variance 4:
 ```python
 X3 = torch.randn(150,20,100) * 2 + 4
 X3 = Variable(X3)
 B2 = nn.BatchNorm1d(20)
 ```

 Note that here we did not set `affine = False`. Instead, we can manually set those values to what we want. To preserve normalization, we want γ(k) = 1 and β(k) = 0. These values are stored in parameters `weight` and `bias` of the BatchNorm variable.

 ```python
 B2.weight.data.fill_(1)
 B2.bias.data.fill_(0)
 Y = B2(X3)

 #Manual calculation
 mu = X3[:,0,:].mean()
 sigma = torch.sqrt(torch.var(X3[:,0,:], unbiased=False) + 1e-5)
 X_normalized = (X3[:,0,:] - mu)/sigma
 ```

 In the above example, `X_normalized` has the same values as `Y[:,0,:]`.

 ### Batch Normalization for images (or any 4D input)

 A batch of RGB images has four dimensions: (B,C,X,Y) or (B,X,Y,C) where B is batch number, C is channel number, and X, Y are locations. In the last example, we only had (B,N) where N was the number of dimensions so it was pretty straightforward to figure out the axis of normalization.

 Along which axis do we normalize here? Normalizing all values across all batch samples is not the solution here. Why? Because each batch sample

 __Hint__: We want to choose an axis that enables a neuron look through all samples in the batch.

 Batch normalization is images is done along the channel axis.

 ```python
X = torch.randn(5,25,100,100) * 2 + 4
X = Variable(X)
B = nn.BatchNorm2d(25, affine=False)
Y = B(X)
 ```

 Here `Y[:,i,:,:]` is the same as
 `((X[:,i,:,:] - X[:,i,:,:].data.mean())/((X[:,i,:,:].data.var(unbiased=False) + 1e-5)**0.5))` for all valid values of `i`.
