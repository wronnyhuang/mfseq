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

<img src="/tex/01a4ed6e24ddd741977314020f649473.svg?invert_in_darkmode&sanitize=true" align=middle width=165.23023439999997pt height=26.76175259999998pt/>H<img src="/tex/25e98a79b8f6c8b0da242a82ff40481e.svg?invert_in_darkmode&sanitize=true" align=middle width=469.4697579pt height=22.831056599999986pt/>W_i<img src="/tex/6e004220e771f29919914a3d71bbdf53.svg?invert_in_darkmode&sanitize=true" align=middle width=237.76756244999993pt height=22.831056599999986pt/>X_i<img src="/tex/5d17741ce0a48f976613e000b90c8641.svg?invert_in_darkmode&sanitize=true" align=middle width=287.66578665pt height=22.831056599999986pt/>H<img src="/tex/6c20aaec28d2560595efb6a807c59e3f.svg?invert_in_darkmode&sanitize=true" align=middle width=700.2745777499999pt height=118.35616319999997pt/>X_i<img src="/tex/fb5220a8732fd491ff94aee8ac7d0760.svg?invert_in_darkmode&sanitize=true" align=middle width=154.80774375pt height=22.831056599999986pt/>\newcommand{\norm}[1]{\left\lVert#1\right\rVert}<img src="/tex/cbe4745c368cb36ecf6b1c81ef1d330a.svg?invert_in_darkmode&sanitize=true" align=middle width=8.21920935pt height=14.15524440000002pt/>L = \sum_i^N \norm{x}<img src="/tex/ef847a88a02d5b2ef1be42a3770000b7.svg?invert_in_darkmode&sanitize=true" align=middle width=181.50738704999998pt height=39.45205439999997pt/>T, W_0, and H<img src="/tex/ef3cb0c949bee696e9d880524395bf99.svg?invert_in_darkmode&sanitize=true" align=middle width=536.48538405pt height=157.8082209pt/>main.py<img src="/tex/1ff258c06ddf59749e2151328bb1abef.svg?invert_in_darkmode&sanitize=true" align=middle width=207.31503045pt height=22.831056599999986pt/>Model.fit<img src="/tex/8ecebfa9de5ea5e0ed6e13e145ce5f74.svg?invert_in_darkmode&sanitize=true" align=middle width=700.2745249499999pt height=203.6529759pt/>forward<img src="/tex/839b0d491281a465798505b53bd36086.svg?invert_in_darkmode&sanitize=true" align=middle width=877.05752475pt height=481.64383740000005pt/>crit$ stands for criterion, which here is the mean-squared-error (MSE) of all the trainable parameters. The training criterion goes way down, indicating that the optimizer is working well. There is a bit of generalization performance, as the test criterion does down about 15%, but better regularizers and more data can make this improve more. The interactive version of these plots are here https://www.comet.ml/wronnyhuang/nmf/e1e9bd13799448f3bf04365e5aec57c4.

![general matrix factorization training curve](doc/traincurve.png)
