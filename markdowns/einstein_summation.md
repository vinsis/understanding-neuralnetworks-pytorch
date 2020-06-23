
### Einstein summation

Einstein summation is a notation to multiply and/or add tensors which can be convenient and intuitive. The posts below do a pretty good job of explaining how it works. This notebook is more of a practice session to get used to applying einstein summation notation.

- [Einstein Summation in Numpy](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/)
- [A basic introduction to NumPy's einsum](https://ajcr.net/Basic-guide-to-einsum/)

The examples here have been taken from the tables in the second link above.


```python
import numpy as np
import torch
```


```python
v = np.arange(6)
# array([0, 1, 2, 3, 4, 5])

w = np.arange(6) + 6
# array([ 6,  7,  8,  9, 10, 11])

A = np.arange(6).reshape(2,3)
# array([[0, 1, 2],
#        [3, 4, 5]])

B = (np.arange(6) + 6).reshape(3,2)
# array([[ 6,  7],
#        [ 8,  9],
#        [10, 11]])

C = np.arange(9).reshape(3,3)
# array([[0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8]])


v_torch = torch.from_numpy(v)
w_torch = torch.from_numpy(w)
A_torch = torch.from_numpy(A)
B_torch = torch.from_numpy(B)
C_torch = torch.from_numpy(C)
```

## 1. Vectors

### 1.1 Return a view of itself


```python
np.einsum('i', v)
```




    array([0, 1, 2, 3, 4, 5])




```python
torch.einsum('i', v_torch)
```




    tensor([0, 1, 2, 3, 4, 5])



### 1.2 Sum up elements of a vector


```python
np.einsum('i->', v)
```




    15




```python
torch.einsum('i->', v_torch)
```




    tensor(15)




```python
v.sum(), v_torch.sum()
```




    (15, tensor(15))



### 1.3 Element-wise operations


```python
np.einsum('i,i->i', v, w)
```




    array([ 0,  7, 16, 27, 40, 55])




```python
v * w
```




    array([ 0,  7, 16, 27, 40, 55])




```python
torch.einsum('i,i->i', v_torch, w_torch)
```




    tensor([ 0,  7, 16, 27, 40, 55])



#### 1.3.1 One can even multiply (element-wise) three or more vectors in the same manner


```python
np.einsum('i,i,i->i', v, w, w)
```




    array([  0,  49, 128, 243, 400, 605])




```python
v*w*w
```




    array([  0,  49, 128, 243, 400, 605])




```python
torch.einsum('i,i,i->i', v_torch, w_torch, w_torch)
```




    tensor([  0,  49, 128, 243, 400, 605])



### 1.4 Inner product


```python
np.einsum('i,i->', v,w)
```




    145




```python
np.dot(v,w)
```




    145




```python
torch.einsum('i,i->',v_torch,w_torch)
```




    tensor(145)



### 1.5 Outer product


```python
np.einsum('i,j->ij',v,w)
```




    array([[ 0,  0,  0,  0,  0,  0],
           [ 6,  7,  8,  9, 10, 11],
           [12, 14, 16, 18, 20, 22],
           [18, 21, 24, 27, 30, 33],
           [24, 28, 32, 36, 40, 44],
           [30, 35, 40, 45, 50, 55]])




```python
np.outer(v,w)
```




    array([[ 0,  0,  0,  0,  0,  0],
           [ 6,  7,  8,  9, 10, 11],
           [12, 14, 16, 18, 20, 22],
           [18, 21, 24, 27, 30, 33],
           [24, 28, 32, 36, 40, 44],
           [30, 35, 40, 45, 50, 55]])




```python
torch.einsum('i,j->ij',v_torch, w_torch)
```




    tensor([[ 0,  0,  0,  0,  0,  0],
            [ 6,  7,  8,  9, 10, 11],
            [12, 14, 16, 18, 20, 22],
            [18, 21, 24, 27, 30, 33],
            [24, 28, 32, 36, 40, 44],
            [30, 35, 40, 45, 50, 55]])



#### 1.5.1 Transpose is just as easy


```python
np.einsum('i,j->ji',v,w)
```




    array([[ 0,  6, 12, 18, 24, 30],
           [ 0,  7, 14, 21, 28, 35],
           [ 0,  8, 16, 24, 32, 40],
           [ 0,  9, 18, 27, 36, 45],
           [ 0, 10, 20, 30, 40, 50],
           [ 0, 11, 22, 33, 44, 55]])



## 2. Matrices

### 2.1 Do nothing


```python
np.einsum('ij',B)
```




    array([[ 6,  7],
           [ 8,  9],
           [10, 11]])




```python
torch.einsum('ij', B_torch)
```




    tensor([[ 6,  7],
            [ 8,  9],
            [10, 11]])



### 2.2 Transpose


```python
A
```




    array([[0, 1, 2],
           [3, 4, 5]])




```python
np.einsum('ij->ji', A)
```




    array([[0, 3],
           [1, 4],
           [2, 5]])




```python
torch.einsum('ij->ji', A_torch)
```




    tensor([[0, 3],
            [1, 4],
            [2, 5]])




```python
print(A.T)
print('---')
print(A_torch.t())
```

    [[0 3]
     [1 4]
     [2 5]]
    ---
    tensor([[0, 3],
            [1, 4],
            [2, 5]])


### 2.3 Diagonal of a square matrix


```python
C
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
np.einsum('ii->i',C)
```




    array([0, 4, 8])




```python
torch.einsum('ii->i', C_torch)
```




    tensor([0, 4, 8])




```python
np.diag(C), torch.diag(C_torch)
```




    (array([0, 4, 8]), tensor([0, 4, 8]))



### 2.4 Trace of a matrix


```python
np.einsum('ii->',C)
```




    12




```python
torch.einsum('ii->', C_torch)
```




    tensor(12)




```python
np.trace(C), torch.trace(C_torch)
```




    (12, tensor(12))



### 2.5 Sum of matrix


```python
np.einsum('ij->',A)
```




    15




```python
torch.einsum('ij->',A_torch)
```




    tensor(15)




```python
A.sum(), A_torch.sum()
```




    (15, tensor(15))



### 2.6 Sum of matrix along axes


```python
np.einsum('ij->j',B)
```




    array([24, 27])




```python
B.sum(0)
```




    array([24, 27])




```python
torch.einsum('ij->j', B_torch)
```




    tensor([24, 27])




```python
torch.einsum('ij->i', B_torch)
```




    tensor([13, 17, 21])




```python
B_torch.sum(1)
```




    tensor([13, 17, 21])



### 2.7 Element-wise multiplication


```python
np.einsum('ij,ij->ij',A,B.T)
```




    array([[ 0,  8, 20],
           [21, 36, 55]])




```python
A * B.T
```




    array([[ 0,  8, 20],
           [21, 36, 55]])



#### 2.7.1 Element-wise multiplication can be done in various ways just by permuting the indices


```python
torch.einsum('ij,ji->ij', C_torch.t(), C_torch)
```




    tensor([[ 0,  9, 36],
            [ 1, 16, 49],
            [ 4, 25, 64]])




```python
C_torch.t() * C_torch.t()
```




    tensor([[ 0,  9, 36],
            [ 1, 16, 49],
            [ 4, 25, 64]])



#### 2.7.2 The below two operations are also the same


```python
torch.einsum('ij,ji->ij', C_torch, C_torch)
```




    tensor([[ 0,  3, 12],
            [ 3, 16, 35],
            [12, 35, 64]])




```python
C_torch * C_torch.t()
```




    tensor([[ 0,  3, 12],
            [ 3, 16, 35],
            [12, 35, 64]])



### 2.8 Matrix multiplication


```python
np.einsum('ij,jk->ik',A,B)
```




    array([[ 28,  31],
           [100, 112]])




```python
np.einsum('ij,jk',A,B)
```




    array([[ 28,  31],
           [100, 112]])




```python
np.dot(A,B)
```




    array([[ 28,  31],
           [100, 112]])




```python
torch.einsum('ij,jk->ik', A_torch, B_torch)
```




    tensor([[ 28,  31],
            [100, 112]])




```python
torch.einsum('ij,jk', A_torch, B_torch)
```




    tensor([[ 28,  31],
            [100, 112]])



### 2.9 Inner product of two matrices


```python
A.shape, B.shape, C.shape
```




    ((2, 3), (3, 2), (3, 3))




```python
print(A)
print('---')
print(C)
```

    [[0 1 2]
     [3 4 5]]
    ---
    [[0 1 2]
     [3 4 5]
     [6 7 8]]



```python
np.einsum('ij,kj->ik', A, C)
```




    array([[ 5, 14, 23],
           [14, 50, 86]])




```python
torch.einsum('ij,kj->ik', A_torch, C_torch)
```




    tensor([[ 5, 14, 23],
            [14, 50, 86]])




```python
i,j = A.shape
k,j2 = C.shape
assert j == j2

result = np.empty((i,k))

for index_i in range(i):
    for index_k in range(k):
        total = 0
        for index_j in range(j):
            total += A[index_i, index_j] * C[index_k, index_j]
        result[index_i, index_k] = total
        
result
```




    array([[ 5., 14., 23.],
           [14., 50., 86.]])



## 3. Higher-order tensors

### 3.1 Each row of A multiplied (element-wise) by C


```python
np.einsum('ij,kj->ikj', A, C)
```




    array([[[ 0,  1,  4],
            [ 0,  4, 10],
            [ 0,  7, 16]],
    
           [[ 0,  4, 10],
            [ 9, 16, 25],
            [18, 28, 40]]])




```python
A
```




    array([[0, 1, 2],
           [3, 4, 5]])




```python
C
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
torch.einsum('ij,kj->ikj', A_torch, C_torch)
```




    tensor([[[ 0,  1,  4],
             [ 0,  4, 10],
             [ 0,  7, 16]],
    
            [[ 0,  4, 10],
             [ 9, 16, 25],
             [18, 28, 40]]])




```python
i,j = A.shape
k,j2 = C.shape
assert j == j2

result = np.empty((i,k,j))

for index_i in range(i):
    for index_k in range(k):
        for index_j in range(j):
            result[index_i, index_k, index_j] = A[index_i, index_j] * C[index_k, index_j]
            
result
```




    array([[[ 0.,  1.,  4.],
            [ 0.,  4., 10.],
            [ 0.,  7., 16.]],
    
           [[ 0.,  4., 10.],
            [ 9., 16., 25.],
            [18., 28., 40.]]])



### 3.2 Each element of first matrix multiplied (element-wise) by second matrix


```python
A
```




    array([[0, 1, 2],
           [3, 4, 5]])




```python
B
```




    array([[ 6,  7],
           [ 8,  9],
           [10, 11]])




```python
np.einsum('ij,kl->ijkl', A, B)
```




    array([[[[ 0,  0],
             [ 0,  0],
             [ 0,  0]],
    
            [[ 6,  7],
             [ 8,  9],
             [10, 11]],
    
            [[12, 14],
             [16, 18],
             [20, 22]]],
    
    
           [[[18, 21],
             [24, 27],
             [30, 33]],
    
            [[24, 28],
             [32, 36],
             [40, 44]],
    
            [[30, 35],
             [40, 45],
             [50, 55]]]])




```python
i,j = A.shape
k,l = B.shape

result = np.empty((i,j,k,l))

for index_i in range(i):
    for index_j in range(j):
        for index_k in range(k):
            for index_l in range(l):
                result[index_i, index_j, index_k, index_l] = A[index_i, index_j] * B[index_k, index_l]
                
result
```




    array([[[[ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.]],
    
            [[ 6.,  7.],
             [ 8.,  9.],
             [10., 11.]],
    
            [[12., 14.],
             [16., 18.],
             [20., 22.]]],
    
    
           [[[18., 21.],
             [24., 27.],
             [30., 33.]],
    
            [[24., 28.],
             [32., 36.],
             [40., 44.]],
    
            [[30., 35.],
             [40., 45.],
             [50., 55.]]]])



## 4. Some examples in PyTorch

### 4.1 Batch multiplication


```python
m1 = torch.randn(4,5,3)
m2 = torch.randn(4,3,5)
torch.bmm(m1,m2)
```




    tensor([[[ 8.2181e-01, -5.8035e-01,  2.2078e+00,  1.4295e+00,  1.8635e+00],
             [ 8.4052e-01, -1.0589e-01,  1.4207e+00,  9.9271e-01,  2.3920e+00],
             [-1.5352e-02, -5.3438e-01,  3.2493e+00,  1.4200e+00,  3.1127e+00],
             [-1.3939e+00,  1.3775e+00, -2.2805e+00, -1.9652e+00,  5.8474e-01],
             [ 2.0536e+00, -6.2420e-01,  2.3070e+00,  2.0755e+00,  2.6713e+00]],
    
            [[ 7.0778e-01,  1.4530e-01,  1.9873e+00,  2.1278e+00, -3.3463e-01],
             [ 1.5298e-01, -1.7556e+00, -2.0336e+00, -3.3895e+00,  2.8165e-03],
             [-4.9915e-01,  2.9698e-02, -9.9656e-01, -5.8711e-01,  3.3424e-01],
             [-7.3538e-02, -6.1606e-01, -1.1962e+00, -1.9709e+00, -1.5120e-02],
             [-8.9944e-01, -2.8037e-01, -2.4217e+00, -2.2450e+00,  5.3909e-01]],
    
            [[ 7.2014e-01,  1.4626e+00, -5.0004e-01,  1.5049e+00, -4.8354e-01],
             [-1.4394e+00,  2.9822e-01,  1.2814e+00,  2.6319e+00,  3.4886e+00],
             [ 4.2215e-01,  1.8446e+00, -5.1419e-03,  1.3402e-01,  1.1296e-01],
             [-2.9111e-01,  7.2741e-01,  1.2039e-01,  4.1217e+00,  1.5959e+00],
             [ 4.1003e-01, -1.7096e-01, -2.7092e-01, -2.1486e+00, -1.2508e+00]],
    
            [[ 8.9113e-01,  1.7029e+00,  1.9923e-01, -3.2940e-01, -9.3853e-01],
             [ 7.0222e+00,  1.5919e-01, -2.6844e+00,  1.0894e+00,  1.2812e+00],
             [ 6.5914e-02, -1.0917e+00, -1.0884e-01,  6.8298e-01,  1.0476e+00],
             [ 3.3019e+00,  2.1760e+00, -1.3713e+00, -1.1536e+00, -1.7116e+00],
             [-4.7698e+00, -1.7244e-01,  1.6248e+00, -9.6778e-01, -1.0414e+00]]])



This is equivalent to:


```python
torch.einsum('bij,bjk->bik', m1, m2)
```




    tensor([[[ 8.2181e-01, -5.8035e-01,  2.2078e+00,  1.4295e+00,  1.8635e+00],
             [ 8.4052e-01, -1.0589e-01,  1.4207e+00,  9.9271e-01,  2.3920e+00],
             [-1.5352e-02, -5.3438e-01,  3.2493e+00,  1.4200e+00,  3.1127e+00],
             [-1.3939e+00,  1.3775e+00, -2.2805e+00, -1.9652e+00,  5.8474e-01],
             [ 2.0536e+00, -6.2420e-01,  2.3070e+00,  2.0755e+00,  2.6713e+00]],
    
            [[ 7.0778e-01,  1.4530e-01,  1.9873e+00,  2.1278e+00, -3.3463e-01],
             [ 1.5298e-01, -1.7556e+00, -2.0336e+00, -3.3895e+00,  2.8165e-03],
             [-4.9915e-01,  2.9698e-02, -9.9656e-01, -5.8711e-01,  3.3424e-01],
             [-7.3538e-02, -6.1606e-01, -1.1962e+00, -1.9709e+00, -1.5120e-02],
             [-8.9944e-01, -2.8037e-01, -2.4217e+00, -2.2450e+00,  5.3909e-01]],
    
            [[ 7.2014e-01,  1.4626e+00, -5.0004e-01,  1.5049e+00, -4.8354e-01],
             [-1.4394e+00,  2.9822e-01,  1.2814e+00,  2.6319e+00,  3.4886e+00],
             [ 4.2215e-01,  1.8446e+00, -5.1419e-03,  1.3402e-01,  1.1296e-01],
             [-2.9111e-01,  7.2741e-01,  1.2039e-01,  4.1217e+00,  1.5959e+00],
             [ 4.1003e-01, -1.7096e-01, -2.7092e-01, -2.1486e+00, -1.2508e+00]],
    
            [[ 8.9113e-01,  1.7029e+00,  1.9923e-01, -3.2940e-01, -9.3853e-01],
             [ 7.0222e+00,  1.5919e-01, -2.6844e+00,  1.0894e+00,  1.2812e+00],
             [ 6.5914e-02, -1.0917e+00, -1.0884e-01,  6.8298e-01,  1.0476e+00],
             [ 3.3019e+00,  2.1760e+00, -1.3713e+00, -1.1536e+00, -1.7116e+00],
             [-4.7698e+00, -1.7244e-01,  1.6248e+00, -9.6778e-01, -1.0414e+00]]])



### The official implementation of [MoCo paper](https://arxiv.org/pdf/1911.05722.pdf) uses `einsum` for calculating dot products with positive and negative examples [as shown here](https://github.com/facebookresearch/moco/blob/master/moco/builder.py#L143-L146):


```python
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
```
