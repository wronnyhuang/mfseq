# Matrix Factorization for Sequential Data

Matrix factorization is used in recommender systems, topic modeling, data compression, and anomaly detection. It is useful for learning latent representations of fixed entities. Sometimes, such entities are changing with respect to time. In such cases, we can extend conventional matrix factorization methods to _sequential_ matrix factorization.

## Formulation
Matrix factorization deals with the following problem.

`X = W * H`

Here we deal with the problem where you have a sequence of data

`X_1, X_2, ..., X_n`

each with a decomposition of

`X_i = W_i * H`

subject to the transition matrix `T`

`W_{i+1} = W_i * T`

`H` is the feature basis, which is consistent over the entire sequence, and `W_i` is the condensed representation of `X_i` on the feature basis spanned by `H`

_Loss_

The loss function for optimization is the frobenius norm of the predicted values of `X_i` with their true values

`L = \sum_i^N distance( X_i, T^(i-1) * W_0 * H )`

The trainable variables are `T, W_0, and H`, which we also l2-regularize to improve generalization.

## Usage

### Requirements
- `tensorflow 1.9` (haven't tested on other versions)
- `numpy`

### Running the code

The code is self contained within the `main.py` file. Toy data is included in the `Model.fit` function. Change this to integrate your data.

A tensorboard of the constructed computation graph is shown here
![tensorboard high level](doc/tensorboard.png)

The main feed-forward computations are enclosed within the `forward` block.
![tensorboard zoom in](doc/tensorboardzoom.png)

