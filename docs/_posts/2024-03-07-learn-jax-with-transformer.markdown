I recently started learning LLM, so wanted to implement [Karpathy's NanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&ab_channel=AndrejKarpathy) using Jax,
with the goal to learn transformer (attention) and Jax at the same time.
Full github repo [here](https://github.com/luffykai/nanogpt-jax).


### Jax "Module" structure

Flax is the neural net library on top of Jax. Similar to `torch.nn.Module`, any layer or model is a `nn.Module`.
A simple model with just linear and activation layers looks like:

```python
from flax import linen as nn

class MLP(nn.Module):
    h_dim: int
    output_dim: int

    def setup(self):
        self.dense1 = nn.Dense(features=self.h_dim)
        self.dense2 = nn.Dense(features=self.output_dim)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.gelu(x)
        x = self.dense2(x)

        return x
``` 

Module is a [dataclass](https://docs.python.org/3/library/dataclasses.html), so no need to define `__init__`.
`setup` defines the layers to be used in `__call__`.

Notice that the linear layer `Dense` only specifies output dimension (features).
This affords flexibility for accommodating different input data dimensions.
However, this also indicates that upon model creation, it is not immediately trainable, given the unknown sizes of tensor variables.

```python
from jax import numpy as jnp

# generates a random tensor.
key = jax.random.PRNGKey(123)
key, x_rng, init_rng = jax.random.split(key, 3)
x = jax.random.uniform(x_rng, shape=(3, 5))

# creates and inits the model
m = MLP(h_dim=10, output_dim=3)
params = model.init(init_rng, x)
```

Ignores the rng things for now (more on it later). Because model doesn't know the 
data dimensions, it's necessary to call `init` with an example data instance (x).
And now we can pass the data through model.

```python
m.apply(params, x)
```

### TrainState

Here is how to actually train the model (updating parameters with gradients)

```python
from flax.training import train_state
import optax

optimizer = optax.adamw(learning_rate=1e-3)
state = train_state.TrainState.create(
    apply_fn=m.apply,
    params=params,
    tx=optimizer,
)

def calculate_loss(state: train_state.TrainState, params, batch):
    x, y = batch
    logits = state.apply_fn(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss

def train_step(state: train_state.TrainState, batch):
    grad_fn = jax.value_and_grad(
        calculate_loss, # takes (state, params, batch)
        argnums=1, # the params is the 1st (0-base) parameter of calculate_loss
    )
    loss, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)

    return state, loss
```

`value_and_grad` returns a functions `grad_fn` that takes the exact same params
as `calculate_loss` (its first param), but computes both loss and gradients

Notice that the model parameters are not "stored inside the model".
The `TrainState` encapsulates all the information:
- the parameters (the matrices for dense1 and dense2)
- the model architecture (model.apply, which is just a function)
- how to update parameters (optimizer)

Initially, this structure might seem counterintuitive when compared to calling `loss.backward()` in torch.
However, this design choice aligns with the functional nature of Jax.
A key principle for functional programming is that functions should be pure (no side effect).
Put differently, calling the same function with identical input yields consistent result.
Consequently, functions (or objects) should refrain from maintaining internal mutable states.
Hence, `params` is managed within `TrainState` outside of the model.
The model merely represents a function defining the architecture.
Notably, each layer also operates as a function(`dense1` is a function that can be invoked on a tensor).

This functional style also explains the RNG shenanigans.

### Jax's RNG

Numpy's RNG works this way:
```python
import numpy as np

np.random.seed(0)
print(np.random.uniform(size=3))
print(np.random.uniform(size=3))
```
These print out two distinct matrices of size 3, which is expected and desired.
The random function is not pure - as repeated invocations yield distinct results.
This suggests the presence of an internal state mananged by the random module.
While this behavior is usually convenient, it has various downsides (complicates debugging for example).
A funcional way to random number generation is to make the state updates explicit.

```python
print(jax.random.uniform(key, shape=(3,)))
print(jax.random.uniform(key, shape=(3,)))
```
The two lines print out the same result, because they have identical state `key`.
To get different random numbers, we always pass in different keys. 
The way to get a new key is to `split`, so now this should make more sense
```python
# generates a random tensor.
key = jax.random.PRNGKey(123)
key, x_rng = jax.random.split(key) # split to get a new key
x = jax.random.uniform(x_rng, shape=(3, 5)) # use the new key (only once)
```

For a more interesting example, let's see how to do dropout.
```python
...
def setup(self):
    ...
    self.attn_drop = nn.Dropout(rate=0.1, deterministic=not self.training)
    ...

def __call__(self):
    ...
    attn = self.attn_drop(attn)
    ...
 ```
 Only adding the layer like above would result in error: *flax.errors.InvalidRngError: attn_drop needs PRNG for "dropout"*.
 Hence it's necessary to supply an rng when executing the model.
 ```python
 def calculate_loss(state: train_state.TrainState, params, batch, rng):
    x, y = batch
    logits = state.apply_fn(params, x, rngs = {"dropout": rng})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss
```

### References
- [UvA DL](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html)
- [Jax 101 on PRNG](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html)
- [Karpathy's video](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&ab_channel=AndrejKarpathy)
