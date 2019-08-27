
Weight normalization was introduced by OpenAI in their paper https://arxiv.org/abs/1602.07868

It is now a part of PyTorch as built-in functionality. It helps in speeding up gradient descent and has been used in Wavenet. 

The basic idea is pretty straight-forward: for each neuron, the weight vector `w` is broken down into two components: its magnitude (`g`) and direction (unit vector in direction of `v`: `v / ||v||`). This we have:

> `w = g * v / ||v||` where `w` and `v` are vectors and `g` is a scalar


Instead of optimizing `w` directly, optimizing on `g` and `v` separately has been found to be faster and more accurate. 


```python
import torch
import torch.nn as nn
```

### There are two things to observe about weight normalization:

1) It increases the number of parameters (by the amount of neurons)

2) It adds a forward pre-hook to the module


```python
get_num_of_params = lambda model: sum([p.numel() for p in model.parameters()])
get_param_name_and_size = lambda model: [(name, param.size()) for (name, param) in model.named_parameters()]
```


```python
linear = nn.Linear(5,3)
```


```python
linear._forward_pre_hooks #no hooks present
```




    OrderedDict()




```python
get_num_of_params(linear), get_param_name_and_size(linear)
```




    (18, [('weight', torch.Size([3, 5])), ('bias', torch.Size([3]))])



### Weight normalization in PyTorch can be done by calling the `nn.utils.weight_norm` function. 

By default, it normalizes the `weight` of a module:


```python
_ = nn.utils.weight_norm(linear)
```

The number of parameters increased by 3 (we have 3 neurons here). Also the parameter `name` is replaced by two parameters `name_g` and `name_v` respectively:


```python
get_num_of_params(linear), get_param_name_and_size(linear)
```




    (21,
     [('bias', torch.Size([3])),
      ('weight_g', torch.Size([3, 1])),
      ('weight_v', torch.Size([3, 5]))])




```python
linear.weight_g.data
```




    tensor([[0.5075],
            [0.4952],
            [0.6064]])




```python
linear.weight_v.data
```




    tensor([[ 0.2782,  0.1382, -0.0764, -0.0963,  0.3820],
            [ 0.0089, -0.2772,  0.0862, -0.3995, -0.0354],
            [ 0.2126,  0.4394, -0.0193,  0.2077, -0.2931]])



Also note that `linear` module now has a forward pre-hook added to it:


```python
linear._forward_pre_hooks
```




    OrderedDict([(0, <torch.nn.utils.weight_norm.WeightNorm at 0x121aac128>)])



The original `weight` is also present but is not a part of `parameters` attribute of `linear` module. Also note that `weight_v.data` is the same as `weight.data`:


```python
linear.weight.data
```




    tensor([[ 0.2782,  0.1382, -0.0764, -0.0963,  0.3820],
            [ 0.0089, -0.2772,  0.0862, -0.3995, -0.0354],
            [ 0.2126,  0.4394, -0.0193,  0.2077, -0.2931]])




```python
linear.weight_v.data
```




    tensor([[ 0.2782,  0.1382, -0.0764, -0.0963,  0.3820],
            [ 0.0089, -0.2772,  0.0862, -0.3995, -0.0354],
            [ 0.2126,  0.4394, -0.0193,  0.2077, -0.2931]])



### How is `weight_g` calculated? We will look at it in a moment


```python
linear.weight_g.data
```




    tensor([[0.5075],
            [0.4952],
            [0.6064]])



### We can get `name` from `name_g` and `name_v` using `torch._weight_norm` function. We will also look at it in a moment


```python
torch._weight_norm(linear.weight_v.data, linear.weight_g.data)
```




    tensor([[ 0.2782,  0.1382, -0.0764, -0.0963,  0.3820],
            [ 0.0089, -0.2772,  0.0862, -0.3995, -0.0354],
            [ 0.2126,  0.4394, -0.0193,  0.2077, -0.2931]])



### Let's look at `norm_except_dim` function

Pretty self-explanatory: it calculates the norm of a tensor except at dimsension provided to the function. We can calculate any Lp norm and omit any dimension we want.

A few examples should make it clear. 


```python
ones = torch.ones(5,5,5)
ones.size()
```




    torch.Size([5, 5, 5])



It we omit dimension `0`, we have 25 elements each of value `1`. Thus their L2 norm is `5`:


```python
norm = 2
dim = 0
y = torch.norm_except_dim(ones, norm, dim)
print('y.size():', y.size())
print('y:', y)
```

    y.size(): torch.Size([5, 1, 1])
    y: tensor([[[5.]],
    
            [[5.]],
    
            [[5.]],
    
            [[5.]],
    
            [[5.]]])


Similar, omitting dim = `0` and calculating L3 norm gives 25 ** (1/3) = 2.9240:


```python
norm = 3
dim = 0
y = torch.norm_except_dim(ones, norm, dim)
print('y.size():', y.size())
print('y:', y)
```

    y.size(): torch.Size([5, 1, 1])
    y: tensor([[[2.9240]],
    
            [[2.9240]],
    
            [[2.9240]],
    
            [[2.9240]],
    
            [[2.9240]]])


Omitting dim = `2` changes the shape of the output:


```python
norm = 2
dim = 2
y = torch.norm_except_dim(ones, norm, dim)
print('y.size():', y.size())
print('y:', y)
```

    y.size(): torch.Size([1, 1, 5])
    y: tensor([[[5., 5., 5., 5., 5.]]])


Omitting dim = `-1` is the same as not omitting anything at all:


```python
torch.norm_except_dim(ones, 2, -1)
```




    tensor(11.1803)




```python
ones.norm()
```




    tensor(11.1803)



### By default `nn.utils.weight_norm` calls `torch.norm_except_dim` with `dim=0`. This is how we get `weight_g`:


```python
torch.norm_except_dim(linear.weight.data, 2, 0)
```




    tensor([[0.5075],
            [0.4952],
            [0.6064]])



It is the same as doing the below operation:


```python
linear.weight.data.norm(dim=1)
```




    tensor([0.5075, 0.4952, 0.6064])



### Let's look at `_weight_norm` function

This function is used to calculate `name` from `name_v` and `name_g`. Let's see how it works:


```python
v = torch.randn(5,3)*10 + 4
g = torch.randn(5,1)
```

For the given values of `g` and `v`, `torch._weight_norm(v,g,0)` is basically the same as `g * v/v.norm(dim=1,keepdim=True)`:


```python
torch._weight_norm(v,g,0)
```




    tensor([[-0.1452, -0.2595, -0.2552],
            [ 0.3853,  0.1218,  0.6896],
            [-0.0612, -0.0946, -0.0837],
            [-0.1722,  0.1708,  0.4013],
            [-0.0354,  0.0114, -0.0197]])




```python
g * v/v.norm(dim=1,keepdim=True)
```




    tensor([[-0.1452, -0.2595, -0.2552],
            [ 0.3853,  0.1218,  0.6896],
            [-0.0612, -0.0946, -0.0837],
            [-0.1722,  0.1708,  0.4013],
            [-0.0354,  0.0114, -0.0197]])



Similarly, `torch._weight_norm(v,g,1)` is basically the same as `g * v/v.norm(dim=0,keepdim=True)`


```python
torch._weight_norm(v,g,1)
```




    tensor([[-0.2282, -0.2974, -0.2645],
            [ 0.3056,  0.0704,  0.3607],
            [-0.0779, -0.0878, -0.0702],
            [-0.0785,  0.0568,  0.1207],
            [-0.0178,  0.0042, -0.0065]])




```python
g * v/v.norm(dim=0,keepdim=True)
```




    tensor([[-0.2282, -0.2974, -0.2645],
            [ 0.3056,  0.0704,  0.3607],
            [-0.0779, -0.0878, -0.0702],
            [-0.0785,  0.0568,  0.1207],
            [-0.0178,  0.0042, -0.0065]])



And `torch._weight_norm(v,g,-1)` is basically the same as `g * v/v.norm()`


```python
torch._weight_norm(v,g,-1)
```




    tensor([[-0.1003, -0.1792, -0.1762],
            [ 0.1343,  0.0424,  0.2403],
            [-0.0342, -0.0529, -0.0468],
            [-0.0345,  0.0342,  0.0804],
            [-0.0078,  0.0025, -0.0043]])




```python
g * v/v.norm()
```




    tensor([[-0.1003, -0.1792, -0.1762],
            [ 0.1343,  0.0424,  0.2403],
            [-0.0342, -0.0529, -0.0468],
            [-0.0345,  0.0342,  0.0804],
            [-0.0078,  0.0025, -0.0043]])



For `linear` module, this is how we get `weight` from `weight_v` and `weight_g` (notice `dim=0`):


```python
torch._weight_norm(linear.weight_v.data, linear.weight_g.data, 0)
```




    tensor([[ 0.2782,  0.1382, -0.0764, -0.0963,  0.3820],
            [ 0.0089, -0.2772,  0.0862, -0.3995, -0.0354],
            [ 0.2126,  0.4394, -0.0193,  0.2077, -0.2931]])




```python
linear.weight.data
```




    tensor([[ 0.2782,  0.1382, -0.0764, -0.0963,  0.3820],
            [ 0.0089, -0.2772,  0.0862, -0.3995, -0.0354],
            [ 0.2126,  0.4394, -0.0193,  0.2077, -0.2931]])



### But what is the point of the forward pre-hook?


```python
linear._forward_pre_hooks
```




    OrderedDict([(0, <torch.nn.utils.weight_norm.WeightNorm at 0x121aac128>)])




```python
hook = linear._forward_pre_hooks[0]
```

#### Let's first see what a hook does: it basically returns the value of `weight.data`.


```python
hook.compute_weight(linear)
```




    tensor([[ 0.2782,  0.1382, -0.0764, -0.0963,  0.3820],
            [ 0.0089, -0.2772,  0.0862, -0.3995, -0.0354],
            [ 0.2126,  0.4394, -0.0193,  0.2077, -0.2931]], grad_fn=<MulBackward0>)



Let's say you are training `linear` on a dataset with `batch_size` = 8. After back-propagation and weight update, the values of `weight_g` and `weight_v` will be different:


```python
batch_size = 8
x = torch.randn(batch_size, 5)
```


```python
y = linear(x)
```


```python
loss = (y-1).sum()
```


```python
loss.backward()
```


```python
linear.weight
```




    tensor([[ 0.2782,  0.1382, -0.0764, -0.0963,  0.3820],
            [ 0.0089, -0.2772,  0.0862, -0.3995, -0.0354],
            [ 0.2126,  0.4394, -0.0193,  0.2077, -0.2931]], grad_fn=<MulBackward0>)




```python
torch._weight_norm(linear.weight_v.data, linear.weight_g.data, 0)
```




    tensor([[ 0.2782,  0.1382, -0.0764, -0.0963,  0.3820],
            [ 0.0089, -0.2772,  0.0862, -0.3995, -0.0354],
            [ 0.2126,  0.4394, -0.0193,  0.2077, -0.2931]])




```python
for param in linear.parameters():
    param.data = param.data - (param.grad.data*0.01)
```

Weights `weight_v` and `weight_g` changed. Hence `weight` should now be equal to:


```python
torch._weight_norm(linear.weight_v.data, linear.weight_g.data, 0)
```




    tensor([[ 0.2946,  0.1498, -0.0491, -0.0748,  0.3667],
            [ 0.0258, -0.2643,  0.1122, -0.3771, -0.0488],
            [ 0.2303,  0.4506,  0.0095,  0.2294, -0.3068]])



But it's not:


```python
linear.weight
```




    tensor([[ 0.2782,  0.1382, -0.0764, -0.0963,  0.3820],
            [ 0.0089, -0.2772,  0.0862, -0.3995, -0.0354],
            [ 0.2126,  0.4394, -0.0193,  0.2077, -0.2931]], grad_fn=<MulBackward0>)



This is why we need a hook. The hook will basically update `linear.weight` by calling `torch._weight_norm(linear.weight_v.data, linear.weight_g.data, 0)` during the next forward propagation:


```python
_ = linear(x)
```

`linear.weight` is updated thanks to the hook:


```python
linear.weight
```




    tensor([[ 0.2946,  0.1498, -0.0491, -0.0748,  0.3667],
            [ 0.0258, -0.2643,  0.1122, -0.3771, -0.0488],
            [ 0.2303,  0.4506,  0.0095,  0.2294, -0.3068]], grad_fn=<MulBackward0>)


