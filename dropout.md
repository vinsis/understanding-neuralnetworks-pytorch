### 1. Theory, motivation and food for thought

The [original paper](http://www.jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) on Dropout is fairly easy to understand and is one of the more interesting research papers I have read. Before I jump into the what and how of Dropout, I think it's important to appreciate _why_ it was needed and what motivated the people involved to come up with this idea.

The below snippet from the original paper gives a good idea:

> Combining several models is most
helpful  when  the  individual  models  are  different  from  each  other  and  in  order  to  make
neural  net  models  different,  they  should  either  have  different  architectures  or  be  trained
on  different  data.   Training  many  different  architectures  is  hard  because  finding  optimal
hyperparameters for each architecture is a daunting task and training each large network
requires a lot of computation.  Moreover, large networks normally require large amounts of
training data and there may not be enough data available to train different networks on
different subsets of the data.  Even if one was able to train many different large networks,
using them all at test time is infeasible in applications where it is important to respond
quickly.

So what is Dropout?

> The  term  “dropout”  refers  to  dropping  out  units  (hidden  and
visible) in a neural network.  By dropping a unit out, we mean temporarily removing it from
the network, along with all its incoming and outgoing connections, as shown in Figure 1.
The choice of which units to drop is random.

__Caveat__: The part `By dropping a unit out, we mean temporarily removing it from
the network, along with all its incoming and outgoing connections` can be a bit misleading. While theoretically correct, it gives an impression that the network architecture changes randomly over time. But the implementation of Dropout does not change the network architecture every iteration. Instead, the output of each node chosen to be dropped out is set to zero regardless of what the input was. This also results in back propagation gradients of weights attached to that node becoming zero (we will soon see this in action).

Before we jump into the code, I would recommend read you read Section 2 of the original paper gives possible explanations for why Dropout works so well. Here's a snippet:

> One possible explanation for the superiority of sexual reproduction is that, over the long
term, the criterion for natural selection may not be individual fitness but rather mix-ability
of genes.  The ability of a set of genes to be able to work well with another random set of
genes makes them more robust.  Since a gene cannot rely on a large set of partners to be
present at all times, it must learn to do something useful on its own or in collaboration with
a
small
number of other genes.  According to this theory, the role of sexual reproduction
is  not  just  to  allow  useful  new  genes  to  spread  throughout  the  population,  but  also  to
facilitate this process by reducing complex co-adaptations that would reduce the chance of
a new gene improving the fitness of an individual.  Similarly, each hidden unit in a neural
network trained with dropout must learn to work with a randomly chosen sample of other
units.  This should make each hidden unit more robust and drive it towards creating useful
features on its own without relying on other hidden units to correct its mistakes.  However,
the hidden units within a layer will still learn to do different things from each other.


### 2. Dropout in action

Think of Dropout as a layer. The output of the layer has the same size (or dimensions) as the input. However, a fraction `p` of the elements in the output is set to zero while the remaining fraction `1-p` of elements are identical (not really but we will see why in a while) with the corresponding input. We can choose any value between (0 and 1 of course) for p.

#### Importing items we need
```python
import torch
import torch.nn as nn
from torch.autograd import Variable
```

#### Create a dropout layer with p = 0.6
```python
p = 0.6
do = nn.Dropout(p)
```

#### Create an input with every element equal to 1
```python
X = torch.ones(5,2)
print(X)
```
Output:
```
1  1
1  1
1  1
1  1
1  1
[torch.FloatTensor of size 5x2]
```

#### Let's see what the output looks like
```python
do(X)
```
Output:
```
Variable containing:
 2.5000  2.5000
 0.0000  2.5000
 2.5000  0.0000
 0.0000  0.0000
 2.5000  2.5000
[torch.FloatTensor of size 5x2]
```

### 2.1 Why?
We see that some of the nodes (__approximately__ a fraction 0.6 have been set of zero in the output). But the remaining ones are not equal to 1.

Why? Because we scaled the input values linearly. Why? Because we want the expected output of each node to be the same with and without dropout.

Why? Because during testing or evaluation dropout is not used.

Why? Because the overall learned network can be thought of as a _combination of several models_ where each _model_ was the result of a dropout layer involved during the training process. Using dropout during testing or evaluation means we are still using a subset of several possible models and not a combination of them which is suboptimal.

#### Rule for scaling the input
Let consider a node `n` with value `x` that is subjected to a dropout layer 100 times. It's output will be `0` approximately `p*100` times and `x_out` `(1-p)*100` times. Its expected output is:

`p*0 + (1-p)*x_out` which is equal to `(1-p)*x_out`

We want it to be equal to x. Hence `x_out = x/(1-p)`

In the example above, `p = 0.6`. Hence, the output is `1/(1-0.6) = 2.5`.

#### Dropout + Linear Layer
Let's create a super small network that looks like this:

```
  Input       Size: (5X10)
    ↓         
 Dropout      p = 0.9
    ↓         
Linear Layer  Size: (10X5)
    ↓         
  Output      (5X5)
```

#### Create input, dropout and linear layer
```python
#input
inputs = torch.ones(5,10)
inputs = Variable(inputs)

#dropout
p = 0.9
do = nn.Dropout(p)

#linear
fc = nn.Linear(10,5)
```

#### Let's look at the output of passing the input through dropout layer:

```python
out = do(inputs)
print(out)
```
Output:
```
Variable containing:
    0     0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     0     0     0     0
    0     0     0     0     0    10     0     0     0     0
    0     0     0    10     0     0     0     0     0     0
    0     0     0     0     0     0     0     0     0     0
[torch.FloatTensor of size 5x10]
```

Since `p = 0.9` about 90% of nodes have been set to 0 in the output. The remaining ones have been scaled by `1/(1-0.9) = 10`.

If you run the above code again and again, the output will be different each time. The output I got by running the above code again is:

```
Variable containing:
    0    10     0     0     0     0     0     0     0     0
    0     0     0     0    10     0     0     0     0     0
    0     0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     0     0     0     0
[torch.FloatTensor of size 5x10]
```

### 2.2 Randomness of sampling subnetworks

Thus, in a way we are sampling random subsets of network by randomly shutting some of the nodes off. But _how random_ are they actually? We see most of the rows tend to be the same (full of zeros). There's not much of randomness going on there.

Let's try setting `p=0.1` and try again:

```python
p = 0.1
do = nn.Dropout(p)
out = do(inputs)
print(out)
```
Output:
```
Variable containing:
 1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  0.0000  1.1111  1.1111
 1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  0.0000  1.1111
 1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  1.1111
 1.1111  1.1111  1.1111  1.1111  0.0000  1.1111  1.1111  1.1111  1.1111  1.1111
 1.1111  0.0000  1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  1.1111  1.1111
[torch.FloatTensor of size 5x10]
```

Run the above code a few times and you will see there's not much of randomness going on. In this case, most rows tend to be full of the same value: `1.1111`.

How do we get maximum randomness in our subnetworks? By setting `p=0.5`. If you are familiar with information theory, we are maximizing entropy here.

```python
p = 0.5
do = nn.Dropout(p)
out = do(inputs)
print(out)
```
Output:
```
Variable containing:
    2     0     2     2     2     2     0     2     2     2
    2     2     2     2     0     2     0     2     2     2
    0     0     2     2     2     2     2     0     2     2
    0     2     2     2     2     2     2     0     0     0
    0     2     2     0     2     2     2     0     0     0
[torch.FloatTensor of size 5x10]
```

It is a standard practice to use dropout with `p=0.5` while training a neural network.

### 2.3 Dropout + Backpropagation

Training a network with dropout layer is pretty straightforward. Quoting from the original paper again:

> Dropout neural networks can be trained using stochastic gradient descent in a manner simi- lar to standard neural nets. The only difference is that for each training case in a mini-batch, we sample a thinned network by dropping out units. Forward and backpropagation for that training case are done only on this thinned network. The gradients for each parameter are averaged over the training cases in each mini-batch. Any training case which does not use a parameter contributes a gradient of zero for that parameter.

Here we put to use the linear layer we created in the last section.

First, let's take a look at the weight and bias parameters of our linear layer:
```python
print(fc.weight)
print(fc.bias)
```
Output:
```
Parameter containing:
 0.0738  0.3089 -0.0683 -0.2662  0.2217 -0.2005  0.0918 -0.0003 -0.0241 -0.0386
-0.0407 -0.1743  0.0462  0.1506  0.2435 -0.2997 -0.1420 -0.1419 -0.1131 -0.2221
-0.0406  0.2344  0.1432 -0.2777 -0.1128  0.0976 -0.1798 -0.0479  0.2498 -0.0814
-0.0947  0.2826 -0.0856  0.2716 -0.1775  0.2035 -0.3161 -0.2716  0.0440 -0.1010
 0.0102 -0.0008 -0.0904  0.2708 -0.0478 -0.1248 -0.0073  0.2026 -0.2273 -0.0355
[torch.FloatTensor of size 5x10]

Parameter containing:
 0.1163
-0.2310
-0.0791
 0.1346
-0.2195
[torch.FloatTensor of size 5]
```

We will create a dummy target and use it to calculate cross entropy loss. We will then use the loss value to backpropagate and take a look at how the gradients look.
```python
target = torch.LongTensor([0,4,2,1,4])
target = Variable(target)
print(target)
```
Output:
```
Variable containing:
 0
 4
 2
 1
 4
[torch.LongTensor of size 5]
```

#### Calculate weight gradients for each training case individually and store in a variable. We use `p=0.9` here.
```python
do = nn.Dropout(0.9)
out = do(inputs)
print(out)
```
Output:
```
Variable containing:
    0    10     0     0     0     0    10     0    10     0
    0     0    10     0     0     0     0     0     0     0
    0    10     0     0     0     0     0    10     0     0
    0     0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     0     0     0     0
[torch.FloatTensor of size 5x10]
```

#### Calculate gradients per training case
We initialize an zero tensor `dWs` and keep adding gradients from each training case to it.
```python
dWs = torch.zeros_like(fc.weight)
for i in range(out.size(0)):
    i_ = out[i].view(1,-1)
    t = target[i]
    o = fc(i_)
    fc.weight.grad.zero_()
    loss = nn.CrossEntropyLoss()(o, t)
    loss.backward()
    dWs += fc.weight.grad
```

#### Let's take a look at the average of `dWs`:
```python
n = dWs.size(0)
print(dWs/n)
```
Output:
```
Variable containing:
 0.0000  0.6987  0.1744  0.0000  0.0000  0.0000 -0.5988  1.2975 -0.5988  0.0000
 0.0000  0.0021  0.3873  0.0000  0.0000  0.0000  0.0003  0.0018  0.0003  0.0000
 0.0000 -1.1258  1.1889  0.0000  0.0000  0.0000  0.5595 -1.6853  0.5595  0.0000
 0.0000  0.1041  0.1495  0.0000  0.0000  0.0000  0.0367  0.0674  0.0367  0.0000
 0.0000  0.3209 -1.9001  0.0000  0.0000  0.0000  0.0022  0.3187  0.0022  0.0000
[torch.FloatTensor of size 5x10]
```

#### Now let's take a look at the gradient of the whole batch. It should be the same as `dWs/n` above.

```python
out = fc(out)
criterion = nn.CrossEntropyLoss()
loss = criterion(out, target)
fc.weight.grad.zero_()
loss.backward()
print(fc.weight.grad)
```
Output:
```
Variable containing:
 0.0000  0.6987  0.1744  0.0000  0.0000  0.0000 -0.5988  1.2975 -0.5988  0.0000
 0.0000  0.0021  0.3873  0.0000  0.0000  0.0000  0.0003  0.0018  0.0003  0.0000
 0.0000 -1.1258  1.1889  0.0000  0.0000  0.0000  0.5595 -1.6853  0.5595  0.0000
 0.0000  0.1041  0.1495  0.0000  0.0000  0.0000  0.0367  0.0674  0.0367  0.0000
 0.0000  0.3209 -1.9001  0.0000  0.0000  0.0000  0.0022  0.3187  0.0022  0.0000
[torch.FloatTensor of size 5x10]
```

We get the same output as `dWs/n`.
