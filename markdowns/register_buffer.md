
## `register_buffer`

The documentation says:

> This is typically used to register a buffer that should not to be considered a model parameter. For example, BatchNorm’s running_mean is not a parameter, but is part of the persistent state.

> Buffers can be accessed as attributes using given names.

As an example we will just implement a simple running average using `register_buffer`:


```python
import torch
import torch.nn as nn
```

`running_average` and `count` are registered as buffers inside the `Model` definition. When an object is created from the `Model` class, it will have `running_average` and `count` as attributed. 

In the `forward` method, you can even define how these attributes will be updated once the model is called.


```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.register_buffer('running_average', torch.Tensor([0.0]))
        self.register_buffer('count', torch.Tensor([0.0]))
        
    def forward(self, x):
        # self.count keeps a count of how many times the model was called
        self.count += 1
        
        # self.running_average keeps the running average in memory
        self.running_average = self.running_average.mul(self.count-1).add(x).div(self.count)
        return x
```

### Note that items registered as buffers are not considered "parameters" of the model. Hence they will not show up under `model.parameters()`:


```python
model = Model()
list(model.parameters())
```




    []



### However they do show up in the `state_dict`. What this means is that when you save the `state_dict`, these values will be saved (and later retrieved) as well:


```python
model.state_dict()
```




    OrderedDict([('running_average', tensor([0.])), ('count', tensor([0.]))])



### Now let's just call the model a couple of times with different values and see how these values change:


```python
model(10)
```




    10




```python
model.count, model.running_average
```




    (tensor([1.]), tensor([10.]))




```python
model(10)
model.count, model.running_average
```




    (tensor([2.]), tensor([10.]))




```python
model(5)
model.count, model.running_average
```




    (tensor([3.]), tensor([8.3333]))




```python
model(15)
model.count, model.running_average
```




    (tensor([4.]), tensor([10.]))




```python
model(1)
model.count, model.running_average
```




    (tensor([5.]), tensor([8.2000]))


