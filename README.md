# Matrix Factorization for Sequential Data

Matrix factorization is used in recommender systems, topic modeling, data compression, and anomaly detection. It is useful for learning latent representations of fixed entities. Sometimes, such entities are changing with respect to time. In such cases, we can extend conventional matrix factorization methods to _sequential_ matrix factorization.

## Formulation
Matrix factorization deals with the following problem.

<img src="/tex/5709ab7abcb2336fed0eb156fd347562.svg?invert_in_darkmode&sanitize=true" align=middle width=69.63454574999999pt height=22.465723500000017pt/>

Here we deal with the problem where you have a sequence of data

<img src="/tex/da31113c64c6c6c46bacbba8c1dd46cd.svg?invert_in_darkmode&sanitize=true" align=middle width=110.30615474999999pt height=22.465723500000017pt/>

each with a decomposition of

<img src="/tex/14efe141a5a9b56763e985b9c0958404.svg?invert_in_darkmode&sanitize=true" align=middle width=77.00709884999998pt height=22.465723500000017pt/>

subject to a transition rule based on transition matrix <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/>

<img src="/tex/f2e64fc75c3319c2122027d1ec872de9.svg?invert_in_darkmode&sanitize=true" align=middle width=168.98686034999997pt height=26.76175259999998pt/>

<img src="/tex/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> is the feature basis, which is consistent over the entire sequence, and <img src="/tex/7185d0c367d394c42432a1246eceab81.svg?invert_in_darkmode&sanitize=true" align=middle width=20.176033349999987pt height=22.465723500000017pt/> is the condensed representation of <img src="/tex/1338d1e5163ba5bc872f1411dd30b36a.svg?invert_in_darkmode&sanitize=true" align=middle width=18.269651399999987pt height=22.465723500000017pt/> on the feature basis spanned by the rows of <img src="/tex/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/>

## Loss Function

The loss function for optimization is the distance (e.g., frobenius norm) of the predicted values of <img src="/tex/1338d1e5163ba5bc872f1411dd30b36a.svg?invert_in_darkmode&sanitize=true" align=middle width=18.269651399999987pt height=22.465723500000017pt/> from their true values, as provided by the data. The trainable variables are <img src="/tex/296143ea793eb8432d75e9086a93abdf.svg?invert_in_darkmode&sanitize=true" align=middle width=61.58575334999999pt height=22.465723500000017pt/>. The optimization objective is the long expression below.

<img src="/tex/aaed2688c2ad47ee82b51f5b87b9c2b1.svg?invert_in_darkmode&sanitize=true" align=middle width=762.87526095pt height=84.34700339999996pt/>

The first term is the *compression* loss, or how much data is lost going from the raw representation to the condensed representation. The second term is the *transition* loss, or how well the transition rule is able to predict the next step, given a compression scheme. The third and fourth terms are weight decay regularizers aimed at getting better generalization. They have a 1-norm in the axis corresponding to the features to enforce sparsity, and a 2-norm in the other axis. The final term is a nonnegativity enforcer, which has zero loss for positive values. The relative importances of each loss term are given by the coefficients <img src="/tex/ae28f809cb0fe8223df4e211f9991f1b.svg?invert_in_darkmode&sanitize=true" align=middle width=60.88275104999999pt height=22.831056599999986pt/>


## Usage

### Requirements
- `tensorflow 1.9` (haven't tested on other versions)
- `numpy`

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
