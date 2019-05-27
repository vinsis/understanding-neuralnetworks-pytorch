

```python
import torch
import torch.nn as nn
```

### Let's create a series of basic operations and calculate gradient


```python
x = torch.tensor([1.,2.,3.], requires_grad=True)
y = x*x
z = torch.sum(y)
z
```




    tensor(14., grad_fn=<SumBackward0>)




```python
y.requires_grad, z.requires_grad
```




    (True, True)




```python
x.grad, y.grad, z.grad
```




    (None, None, None)




```python
z.backward()
```


```python
x.grad, y.grad, z.grad
```




    (tensor([2., 4., 6.]), None, None)



### You can register a hook to a PyTorch `Module` or `Tensor`


```python
x = torch.tensor([1.,2.,3.], requires_grad=True)
y = x*x
z = torch.sum(y)
z
```




    tensor(14., grad_fn=<SumBackward0>)



#### `register_hook` registers a backward hook


```python
h = y.register_hook(lambda grad: print(grad))
```


```python
z.backward()
```

    tensor([1., 1., 1.])



```python
x.grad, y.grad, z.grad
```




    (tensor([2., 4., 6.]), None, None)



#### You can also use a hook to manipulte gradients. The hook should have the following signature:

`hook(grad) -> Tensor or None`


```python
x = torch.tensor([1.,2.,3.], requires_grad=True)
y = x*x
z = torch.sum(y)
z
```




    tensor(14., grad_fn=<SumBackward0>)



#### Let's add random noise to gradient of `y`


```python
def add_random_noise_and_print(grad):
    noise = torch.randn(grad.size())
    print('Noise:', noise)
    return grad + noise
    
h = y.register_hook(add_random_noise_and_print)
```


```python
x.grad, y.grad, z.grad
```




    (None, None, None)




```python
z.backward()
```

    Noise: tensor([1.1173, 1.2854, 0.0611])



```python
x.grad, y.grad, z.grad
```




    (tensor([4.2346, 9.1415, 6.3665]), None, None)



### Forward and backward hooks

There are three hooks listed below. They are available only for `nn.Module`s

* __`register_forward_pre_hook`__: function is called BEFORE forward call
    * Signature: `hook(module, input) -> None
    
    
* __`register_forward_hook`__: function is called AFTER forward call
    * Signature: `hook(module, input, output) -> None`
    
    
* __`register_backward_hook`__: function is called AFTER gradients wrt module input are computed
    * Signature: `hook(module, grad_input, grad_output) -> Tensor or None`

#### `register_forward_pre_hook`


```python
linear = nn.Linear(10,1)
```


```python
[param for param in linear.parameters()]
```




    [Parameter containing:
     tensor([[-0.3009,  0.0351, -0.2786,  0.1136, -0.2712,  0.0183, -0.2881, -0.1555,
              -0.3108,  0.0767]], requires_grad=True), Parameter containing:
     tensor([0.0377], requires_grad=True)]




```python
h = linear.register_forward_pre_hook(lambda _, inp: print(inp))
```


```python
x = torch.ones([8,10])
```


```python
y = linear(x)
```

    (tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]),)


#### `register_forward_hook`


```python
linear = nn.Linear(10,1)
[param for param in linear.parameters()]
```




    [Parameter containing:
     tensor([[ 0.0978, -0.1878,  0.0189,  0.3040,  0.1120,  0.1977,  0.2137, -0.2841,
              -0.0718,  0.2079]], requires_grad=True), Parameter containing:
     tensor([0.2796], requires_grad=True)]




```python
h = linear.register_forward_hook(lambda _, inp, out: print(inp, '\n\n', out))
```


```python
y = linear(x)
```

    (tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]),) 
    
     tensor([[0.8878],
            [0.8878],
            [0.8878],
            [0.8878],
            [0.8878],
            [0.8878],
            [0.8878],
            [0.8878]], grad_fn=<AddmmBackward>)


Just to verify, the result above can also be computed manually like so:


```python
[param for param in linear.parameters()][0].sum() + [param for param in linear.parameters()][1]
```




    tensor([0.8878], grad_fn=<AddBackward0>)



#### `register_backward_hook`


```python
linear = nn.Linear(3,1)
[param for param in linear.parameters()]
```




    [Parameter containing:
     tensor([[0.5395, 0.2303, 0.5583]], requires_grad=True), Parameter containing:
     tensor([-0.3510], requires_grad=True)]




```python
def print_sizes(module, grad_inp, grad_out):
    print(len(grad_inp), len(grad_out))
    print('===')
    print('Grad input sizes:', [i.size() for i in grad_inp if i is not None])
    print('Grad output sizes:', [i.size() for i in grad_out if i is not None])
    print('===')
    print('Grad_input 0:', grad_inp[0])
    print('Grad_input 1:', grad_inp[1])
    print('Grad_input 2:', grad_inp[2])
    print('===')
    print(grad_out)

h = linear.register_backward_hook(print_sizes)
```


```python
x = torch.ones([8,3]) * 10
y = linear(x)
```


```python
y.backward(torch.ones_like(y) * 1.5)
```

    3 1
    ===
    Grad input sizes: [torch.Size([1]), torch.Size([3, 1])]
    Grad output sizes: [torch.Size([8, 1])]
    ===
    Grad_input 0: tensor([12.])
    Grad_input 1: None
    Grad_input 2: tensor([[120.],
            [120.],
            [120.]])
    ===
    (tensor([[1.5000],
            [1.5000],
            [1.5000],
            [1.5000],
            [1.5000],
            [1.5000],
            [1.5000],
            [1.5000]]),)


`register_backward_hook` can be used to tweak `grad_inp` values after they have been calculated. To do so, the hook function should return <i>a new gradient with respect to input that will be used in place of `grad_input` in subsequent computations</i>

In the example below, we add random noise to the gradient of bias using the hook:


```python
linear = nn.Linear(3,1)
[param for param in linear.parameters()]
```




    [Parameter containing:
     tensor([[-0.5438,  0.5539,  0.5210]], requires_grad=True),
     Parameter containing:
     tensor([-0.4839], requires_grad=True)]




```python
def add_noise_and_print(module, grad_inp, grad_out):
    noise = torch.randn(grad_inp[0].size())
    print('Noise:', noise)
    print('===')
    print('Grad input sizes:', [i.size() for i in grad_inp if i is not None])
    print('Grad output sizes:', [i.size() for i in grad_out if i is not None])
    print('===')
    print('Grad_input 0:', grad_inp[0])
    print('Grad_input 1:', grad_inp[1])
    print('Grad_input 2:', grad_inp[2])
    print('===')
    print(grad_out)
    return (grad_inp[0] + noise, None, grad_inp[2])

h = linear.register_backward_hook(add_noise_and_print)
```


```python
x = torch.ones([8,3]) * 10
y = linear(x)
```


```python
y.backward(torch.ones_like(y) * 1.5)
```

    Noise: tensor([0.7553])
    ===
    Grad input sizes: [torch.Size([1]), torch.Size([3, 1])]
    Grad output sizes: [torch.Size([8, 1])]
    ===
    Grad_input 0: tensor([12.])
    Grad_input 1: None
    Grad_input 2: tensor([[120.],
            [120.],
            [120.]])
    ===
    (tensor([[1.5000],
            [1.5000],
            [1.5000],
            [1.5000],
            [1.5000],
            [1.5000],
            [1.5000],
            [1.5000]]),)


`linear.bias.grad` was originally `12.0` but a random noise `Noise: tensor([0.7553])` was added to it:

```
12 + 0.7553 = 12.7553
```


```python
linear.bias.grad
```




    tensor([12.7553])




```python
linear.weight.grad
```




    tensor([[120., 120., 120.]])


