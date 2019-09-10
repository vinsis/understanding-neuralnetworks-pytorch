
### Solve the same problem in different ways

__Problem__: Find square root of `5`.

__Actual solutions__: `±2.2360679775`

### 1. Start with a guess and improve the guess using gradient descent on loss


```python
import torch
```


```python
guess = torch.tensor(4.0, requires_grad=True)
learning_rate = 0.01
```


```python
for _ in range(20):
    loss = torch.pow(5 - torch.pow(guess, 2), 2)
    if guess.grad is not None:
        guess.grad.data.zero_()
    loss.backward()
    guess.data = guess.data - learning_rate * guess.grad.data
    print(guess.item())
```

    2.240000009536743
    2.2384231090545654
    2.2374794483184814
    2.2369143962860107
    2.2365756034851074
    2.236372470855713
    2.236250638961792
    2.236177682876587
    2.2361338138580322
    2.236107587814331
    2.2360918521881104
    2.2360823154449463
    2.236076593399048
    2.2360732555389404
    2.2360711097717285
    2.236069917678833
    2.2360692024230957
    2.2360687255859375
    2.2360684871673584
    2.2360682487487793


### 2. Parameterize the guess: `shift` the guess

The idea here is that you do not change the input parameter. Instead, you come up with some variables which interact with the input to create a guess. The variables are then updated using gradient descent. These variables are also called `parameters`.

Here we use a parameter called `shift`.

__This is an import shift (no pun intended) in how you think about getting the right answers__: 
> __Don't update the guess. Instead update the variables that interact with the guess. A similar line of thought employed in reparameterization trick used in variational inference on neuron networks.__


```python
input_val = torch.tensor(4.0)
shift = torch.tensor(1.0, requires_grad=True)
```


```python
for _ in range(30):
    guess = input_val + shift
    loss = torch.pow(5 - torch.pow(guess, 2), 2)
    if shift.grad is not None:
        shift.grad.data.zero_()
    loss.backward()
    shift.data = shift.data - learning_rate * shift.grad.data
    print(guess.item())
```

    5.0
    1.0
    1.1600000858306885
    1.3295643329620361
    1.5014641284942627
    1.6663613319396973
    1.8145501613616943
    1.9384772777557373
    2.034804582595825
    2.104766845703125
    2.152751922607422
    2.184238910675049
    2.2042551040649414
    2.216710090637207
    2.2243528366088867
    2.2290022373199463
    2.2318150997161865
    2.233511447906494
    2.234532356262207
    2.2351458072662354
    2.2355144023895264
    2.2357358932495117
    2.235868453979492
    2.235948324203491
    2.2359962463378906
    2.236024856567383
    2.236042022705078
    2.2360525131225586
    2.2360587120056152
    2.236062526702881


### 2.1 Parameterize the guess: `shift` and `scale` the guess


```python
input_val = torch.tensor(4.0)
shift = torch.tensor(1.0, requires_grad=True)
scale = torch.tensor(1.0, requires_grad=True)
```

The learning rate of 0.01 is pretty high for this operation. Thus we reduce it (in log scale). After a couple of runs, we decide on the rate of `0.0005`.


```python
learning_rate = 0.0005
for _ in range(30):
    guess = input_val*scale + shift
    loss = torch.pow(5 - torch.pow(guess, 2), 2)
    if shift.grad is not None:
        shift.grad.data.zero_()
    if scale.grad is not None:
        scale.grad.data.zero_()
    loss.backward()
    shift.data = shift.data - learning_rate * shift.grad.data
    scale.data = scale.data - learning_rate * scale.grad.data
    print(guess.item())
```

    5.0
    1.5999999046325684
    1.7327359914779663
    1.8504221439361572
    1.9495712518692017
    2.0290589332580566
    2.0899698734283447
    2.134881019592285
    2.1669845581054688
    2.1893954277038574
    2.204770803451538
    2.2151894569396973
    2.222188949584961
    2.2268640995025635
    2.2299740314483643
    2.2320375442504883
    2.2334041595458984
    2.2343082427978516
    2.234905958175659
    2.2353007793426514
    2.2355613708496094
    2.2357337474823
    2.235847234725952
    2.235922336578369
    2.2359719276428223
    2.23600435256958
    2.2360260486602783
    2.2360403537750244
    2.2360496520996094
    2.236055850982666


### 2.3 Note that the loss function has two minima: with a little bit of tweaking, a relatively higher learning rate pushes it to another minimum.

Another interesting thing to note is that there are two solutions to square root of 5: `2.2360` and `-2.2360`. If we increase the learning rate to `0.001`, it pushes it into another minima. The solution converges to `-2.2360680103302`.


```python
input_val = torch.tensor(4.0)
shift = torch.tensor(1.0, requires_grad=True)
scale = torch.tensor(1.0, requires_grad=True)

learning_rate = 0.001
for _ in range(30):
    guess = input_val*scale + shift
    loss = torch.pow(5 - torch.pow(guess, 2), 2)
    if shift.grad is not None:
        shift.grad.data.zero_()
    if scale.grad is not None:
        scale.grad.data.zero_()
    loss.backward()
    shift.data = shift.data - learning_rate * shift.grad.data
    scale.data = scale.data - learning_rate * scale.grad.data
    print(guess.item())
```

    5.0
    -1.8000000715255737
    -2.0154242515563965
    -2.1439850330352783
    -2.202786445617676
    -2.2249152660369873
    -2.2324423789978027
    -2.2349019050598145
    -2.235694408416748
    -2.235948085784912
    -2.236029624938965
    -2.236055850982666
    -2.2360641956329346
    -2.2360668182373047
    -2.236067533493042
    -2.236067771911621
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302
    -2.2360680103302


### 3. Replacing `scale` and `shift` with a neuron


```python
neuron = torch.nn.Linear(1,1)
input_val = torch.tensor(4.0).view(1,-1) # input to a linear layer should be in the form [batch_size, input_size]

learning_rate = 0.001
for _ in range(30):
    guess = neuron(input_val)
    loss = torch.pow(5 - torch.pow(guess, 2), 2)
    neuron.zero_grad()
    loss.backward()
    for param in neuron.parameters(): # parameters of a model can be iterated through easily using the `.parameters()` method
        param.data = param.data - learning_rate * param.grad.data
    print(guess.item())
```

    0.2342168092727661
    0.3129768371582031
    0.41730424761772156
    0.5542460680007935
    0.7311122417449951
    0.9531161785125732
    1.2182986736297607
    1.5095584392547607
    1.7888929843902588
    2.0078368186950684
    2.1400814056396484
    2.201209545135498
    2.2243618965148926
    2.232259511947632
    2.234842538833618
    2.23567533493042
    2.2359421253204346
    2.236027717590332
    2.2360548973083496
    2.2360639572143555
    2.2360665798187256
    2.236067533493042
    2.236067771911621
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302


### 5. Using the optimizer to do parameter updates for you 

We tell the optimizer two things: 
a) what paramters need to be updated
b) what is the learning rate

Then it does all the hard work for you:


```python
neuron = torch.nn.Linear(1,1)
optimizer = torch.optim.SGD(neuron.parameters(), lr=0.001)

input_val = torch.tensor(4.0).view(1,-1) # input to a linear layer should be in the form [batch_size, input_size]

learning_rate = 0.001
for _ in range(30):
    guess = neuron(input_val)
    loss = torch.pow(5 - torch.pow(guess, 2), 2)
    neuron.zero_grad()
    loss.backward()
    optimizer.step()
    print(guess.item())
```

    1.4186859130859375
    1.7068755626678467
    1.949059247970581
    2.108257293701172
    2.187859058380127
    2.2195885181427
    2.230670928955078
    2.234327793121338
    2.2355096340179443
    2.235889196395874
    2.236010789871216
    2.2360496520996094
    2.2360620498657227
    2.2360661029815674
    2.236067295074463
    2.236067771911621
    2.236067771911621
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302


### 6. Using the loss function to calculate the loss automatically

In some cases calculating the loss may not be as trivial as it is in this case. In those scenarios, we can use PyTorch's in-built loss functions.

Here we will be using `MSELoss`:


```python
neuron = torch.nn.Linear(1,1)
optimizer = torch.optim.SGD(neuron.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss()

input_val = torch.tensor(4.0).view(1,-1) # input to a linear layer should be in the form [batch_size, input_size]

learning_rate = 0.001
for _ in range(30):
    guess = neuron(input_val)
    predicted_output = torch.pow(guess, 2)
    actual_output = torch.tensor(5.0).view(1,-1)
    loss = loss_function(predicted_output, actual_output)
    neuron.zero_grad()
    loss.backward()
    optimizer.step()
    print(guess.item())
```

    3.8363184928894043
    1.3013595342636108
    1.593956470489502
    1.860517978668213
    2.0551581382751465
    2.1636502742767334
    2.2105278968811035
    2.2275986671447754
    2.2333250045776367
    2.235186815261841
    2.235785722732544
    2.2359776496887207
    2.236039161682129
    2.2360587120056152
    2.236064910888672
    2.236067295074463
    2.236067533493042
    2.236067771911621
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302
    2.2360680103302

