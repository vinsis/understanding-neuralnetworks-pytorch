

```python
import torch
import torch.nn as nn
```

### 1. LogSoftmax


```python
x = torch.randn(8, 5) 
nn.LogSoftmax(dim=1)(x)
```




    tensor([[-3.3124, -2.3836, -1.1339, -1.4844, -1.1303],
            [-1.9481, -1.3503, -2.9377, -1.4468, -1.1712],
            [-2.1393, -1.8147, -4.4871, -1.4645, -0.7404],
            [-3.1521, -2.6411, -1.9489, -2.7951, -0.3821],
            [-1.6078, -1.3891, -2.0410, -3.2716, -0.9610],
            [-1.2204, -2.6508, -1.1108, -2.0820, -1.7130],
            [-2.0168, -0.9498, -1.7237, -1.5997, -2.3051],
            [-2.3434, -1.8218, -2.2064, -0.5169, -3.3294]])



### Same as taking the softmax first and then taking log of each value
* Softmax: `x.exp() / x.exp().sum(dim=1, keepdim=True)`
* Log: `().log()`


```python
(x.exp() / x.exp().sum(dim=1, keepdim=True)).log()
```




    tensor([[-3.3124, -2.3836, -1.1339, -1.4844, -1.1303],
            [-1.9481, -1.3503, -2.9377, -1.4468, -1.1712],
            [-2.1393, -1.8147, -4.4871, -1.4645, -0.7404],
            [-3.1521, -2.6411, -1.9489, -2.7951, -0.3821],
            [-1.6078, -1.3891, -2.0410, -3.2716, -0.9610],
            [-1.2204, -2.6508, -1.1108, -2.0820, -1.7130],
            [-2.0168, -0.9498, -1.7237, -1.5997, -2.3051],
            [-2.3434, -1.8218, -2.2064, -0.5169, -3.3294]])



### 2. NLLLoss

> The negative log likelihood loss. It is useful to train a classification problem with C classes.<br>
If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes. This is particularly useful when you have an unbalanced training set.

### Simply put, (log of probability * -1)


```python
targets = torch.tensor([4,2,3,4,0,0,1,2])
softmax_x = x.exp() / x.exp().sum(dim=1,keepdim=True)
log_likelihood_x = softmax_x.log()
print(log_likelihood_x.size(), targets.size())
nn.NLLLoss()(log_likelihood_x, targets)
```

    torch.Size([8, 5]) torch.Size([8])





    tensor(1.4874)




```python
nn.NLLLoss()(nn.LogSoftmax(dim=1)(x), targets)
```




    tensor(1.4874)



### Manual calculation


```python
loss = 0
for i in range(8):
    loss += nn.LogSoftmax(dim=1)(x)[i][targets[i]]
    
loss/8
```




    tensor(-1.4874)



### 3. CrossEntropyLoss

This criterion combines `nn.LogSoftmax()` and `nn.NLLLoss()` in one single class.


```python
nn.CrossEntropyLoss()(x, targets)
```




    tensor(1.4874)



### 4. BCELoss

> Creates a criterion that measures the Binary Cross Entropy between the target and the output.

`CrossEntropyLoss` measures how close is the probability of true class close to 1. It does not consider what other possibilities are. Thus, it works well with `Softmax`.

When you want to measure how close are the probabilities of true classes close to 1 and how close are the probabilities of non-true classes close to 0, using `BCELoss` makes sense.


```python
x = (torch.randn(4,3)).sigmoid_()
x
```




    tensor([[0.4929, 0.8562, 0.5976],
            [0.3183, 0.7167, 0.5629],
            [0.4342, 0.5078, 0.7811],
            [0.6997, 0.6790, 0.7381]])




```python
targets = torch.FloatTensor([[0,1,1],[1,1,0],[0,0,0],[1,0,0]])
targets
```




    tensor([[0., 1., 1.],
            [1., 1., 0.],
            [0., 0., 0.],
            [1., 0., 0.]])




```python
nn.BCELoss()(x, targets)
```




    tensor(0.7738)



### Manual calculation


```python
def loss_per_class(p,t):
    if t == 0:
        return -1 * (1-t) * torch.log(1-p)
    else:
        return -1 * t * torch.log(p)

loss = 0
for index in range(x.size(0)):
    predicted = x[index]
    true = targets[index]
    loss += torch.FloatTensor([loss_per_class(p,t) for p,t in zip(predicted, true)]).sum()
    
loss / (4*3)
```




    tensor(0.7738)



### 5. KLDivLoss

As with `NLLLoss`, the input given is expected to contain _log-probabilities_. However, unlike `NLLLoss`, input is not restricted to a 2D Tensor, because the criterion is applied element-wise. The targets are given as _probabilities_ (i.e. without taking the logarithm).

This criterion expects a target Tensor of the same size as the input Tensor.


```python
x = torch.randn(8,5)
targets = torch.randn_like(x).sigmoid_()
nn.KLDivLoss()(nn.LogSoftmax(dim=1)(x), targets)
```




    tensor(0.8478)



### Manual calculation


```python
loss = 0
for i in range(x.size(0)):
    for j in range(x.size(1)):
        loss += targets[i,j] * (torch.log(targets[i,j]) - x[i,j] + x[i,:].exp().sum().log())
```


```python
nn.KLDivLoss(reduction='sum')(nn.LogSoftmax(dim=1)(x), targets), loss 
```




    (tensor(33.9101), tensor(33.9101))




```python
nn.KLDivLoss()(nn.LogSoftmax(dim=1)(x), targets), loss / (x.size(0) * x.size(1)) 
```




    (tensor(0.8478), tensor(0.8478))


