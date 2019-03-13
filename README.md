# Matrix Factorization for Sequential Data

Matrix factorization is used in recommender systems, topic modeling, data compression, and anomaly detection. It is useful for learning latent representations of fixed entities. Sometimes, such entities are changing with respect to time. In such cases, we can extend conventional matrix factorization methods to _sequential_ matrix factorization.

## Formulation
Matrix factorization deals with the following problem.

<img src="/tex/5709ab7abcb2336fed0eb156fd347562.svg?invert_in_darkmode&sanitize=true" align=middle width=69.63454574999999pt height=22.465723500000017pt/>

Here we deal with the problem where you have a sequence of data

<img src="/tex/257607c8a2084cbcd56bd4af1d918051.svg?invert_in_darkmode&sanitize=true" align=middle width=99.34751804999999pt height=22.465723500000017pt/>

each with a decomposition of

<img src="/tex/a8505c0da8435483beafa5e5e28d4a5b.svg?invert_in_darkmode&sanitize=true" align=middle width=92.53206599999999pt height=22.465723500000017pt/>

subject to the transition matrix <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/>

<img src="/tex/66e779f879f103e83e2877c5fd8174a4.svg?invert_in_darkmode&sanitize=true" align=middle width=107.97168194999999pt height=22.465723500000017pt/>

<img src="/tex/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> is the feature basis, which is consistent over the entire sequence, and <img src="/tex/7185d0c367d394c42432a1246eceab81.svg?invert_in_darkmode&sanitize=true" align=middle width=20.176033349999987pt height=22.465723500000017pt/> is the condensed representation of <img src="/tex/1338d1e5163ba5bc872f1411dd30b36a.svg?invert_in_darkmode&sanitize=true" align=middle width=18.269651399999987pt height=22.465723500000017pt/> on the feature basis spanned by <img src="/tex/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/>

We make the optimization easier by calculating only the residual of each transition

<img src="/tex/0cfc064e4ec5ca0c19f64675af7b30f1.svg?invert_in_darkmode&sanitize=true" align=middle width=169.4166144pt height=22.465723500000017pt/>

_Loss_

The loss function for optimization is the distance (e.g., frobenius norm) of the predicted values of <img src="/tex/1338d1e5163ba5bc872f1411dd30b36a.svg?invert_in_darkmode&sanitize=true" align=middle width=18.269651399999987pt height=22.465723500000017pt/> from their true values

<img src="/tex/4cbfc855b154e568b3437911ab4a245e.svg?invert_in_darkmode&sanitize=true" align=middle width=293.19592005pt height=32.256008400000006pt/>

The trainable variables are <img src="/tex/4ab841e2c685781e6d9177d6b28b3ae7.svg?invert_in_darkmode&sanitize=true" align=middle width=90.59940945pt height=22.831056599999986pt/>, which we also l2-regularize to improve generalization.

## Usage

### Requirements
- <img src="/tex/b5970b03abf71a1c8c1ebedac53a2a73.svg?invert_in_darkmode&sanitize=true" align=middle width=103.23289514999998pt height=22.831056599999986pt/> (haven't tested on other versions)
- <img src="/tex/8da05e172086a73d2e1e8164797669f3.svg?invert_in_darkmode&sanitize=true" align=middle width=50.63004209999999pt height=14.15524440000002pt/>

### Running the code

The code is self contained within the <img src="/tex/24f3b5d61559402ec87940f65513fb36.svg?invert_in_darkmode&sanitize=true" align=middle width=60.13837334999999pt height=21.68300969999999pt/> file. Toy data is included in the <img src="/tex/43af940070a675cffcce616b979f3bd6.svg?invert_in_darkmode&sanitize=true" align=middle width=73.12917314999999pt height=22.831056599999986pt/> function. Change this to integrate your data.

### Computation graph visualization
An interactive tensorboard of the constructed computation graph is in the following link
https://boards.aughie.org/board/IQg3vzIEAcHviNIniDicLLQ-U_E/

Screenshot of high level graph

![tensorboard high level](doc/tensorboard.png)

The main feed-forward computations are enclosed within the <img src="/tex/6a5854152f3d92fa90cd1c442142c66a.svg?invert_in_darkmode&sanitize=true" align=middle width=62.98733759999998pt height=22.831056599999986pt/> block.

![tensorboard zoom in](doc/tensorboardzoom.png)

## Results on graph data 

### Data exploration and normalization

Typically graph features (such as those extracted by ReFeX) can contain large variations in the values. To make the optimization work, we must first normalize the graph features values. To do this, we mean-standardize each feature's marginal distribution. We show the data exploration and normalization process in this notebook https://colab.research.google.com/drive/13Gj8qYA2Nl8jucQbWaYeuclmaStkwODL

A gif of all the nonzero feature marginal distributions is shown here. The data comes from features extraction from a graph of transactions at a financial institution.

![histograms](doc/histograms.gif)


### Results

General (negative values allowed) matrix factorization yields the following training curve. <img src="/tex/75528dd9fe475ae921d58755998e8e40.svg?invert_in_darkmode&sanitize=true" align=middle width=26.58608369999999pt height=21.68300969999999pt/> stands for criterion, which here is the mean-squared-error (MSE) of all the trainable parameters. The training criterion goes way down, indicating that the optimizer is working well. There is a bit of generalization performance, as the test criterion does down about 15%, but better regularizers and more data can make this improve more. The interactive version of these plots are here https://www.comet.ml/wronnyhuang/nmf/e1e9bd13799448f3bf04365e5aec57c4.

![general matrix factorization training curve](doc/traincurve.png)
