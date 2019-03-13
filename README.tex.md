# Matrix Factorization for Sequential Data

Matrix factorization is used in recommender systems, topic modeling, data compression, and anomaly detection. It is useful for learning latent representations of fixed entities. Sometimes, such entities are changing with respect to time. In such cases, we can extend conventional matrix factorization methods to _sequential_ matrix factorization.

## Formulation
Matrix factorization deals with the following problem.

$X = WH$

Here we deal with the problem where you have a sequence of data

$X_1, X_2, \cdots, X_n$

each with a decomposition of

$X_i = W_i H$

subject to a transition rule based on transition matrix $T$

$W_i = W_{i-1}T + w_{i-2}T^2$

$H$ is the feature basis, which is consistent over the entire sequence, and $W_i$ is the condensed representation of $X_i$ on the feature basis spanned by the rows of $H$

**Loss**

The loss function for optimization is the distance (e.g., frobenius norm) of the predicted values of $X_i$ from their true values

$L = \alpha\sum_{i=1}^N \left\lVert W_i H - X_i \right\rVert^2 + 
(1-\alpha)\sum_{i=3}^N \left\lVert W_i H - (W_{i-2}T^2H + W_i T H) \right\rVert^2 + 
\beta\sum_{i=1}^N \left\lVert W_i \right\rVert_{2,1}^2 + 
\kappa\left\lVert H \right\rVert_{1,2}^2 + 
\eta\left\lVert \mathrm{relu(all W and H elements)} \right\rVert^2 $

The trainable variables are $T, W_0, and H$, which we also l2-regularize to improve generalization.

## Usage

### Requirements
- `tensorflow 1.9` (haven't tested on other versions)
- `numpy`

### Running the code

The code is self contained within the $main.py$ file. Toy data is included in the $Model.fit$ function. Change this to integrate your data.

### Computation graph visualization
An interactive tensorboard of the constructed computation graph is in the following link
https://boards.aughie.org/board/IQg3vzIEAcHviNIniDicLLQ-U_E/

Screenshot of high level graph

![tensorboard high level](doc/tensorboard.png)

The main feed-forward computations are enclosed within the $forward$ block.

![tensorboard zoom in](doc/tensorboardzoom.png)

## Results on graph data 

### Data exploration and normalization

Typically graph features (such as those extracted by ReFeX) can contain large variations in the values. To make the optimization work, we must first normalize the graph features values. To do this, we mean-standardize each feature's marginal distribution. We show the data exploration and normalization process in this notebook https://colab.research.google.com/drive/13Gj8qYA2Nl8jucQbWaYeuclmaStkwODL

A gif of all the nonzero feature marginal distributions is shown here. The data comes from features extraction from a graph of transactions at a financial institution.

![histograms](doc/histograms.gif)


### Results

General (negative values allowed) matrix factorization yields the following training curve. $crit$ stands for criterion, which here is the mean-squared-error (MSE) of all the trainable parameters. The training criterion goes way down, indicating that the optimizer is working well. There is a bit of generalization performance, as the test criterion does down about 15%, but better regularizers and more data can make this improve more. The interactive version of these plots are here https://www.comet.ml/wronnyhuang/nmf/e1e9bd13799448f3bf04365e5aec57c4.

![general matrix factorization training curve](doc/traincurve.png)
