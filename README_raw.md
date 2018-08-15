# Gaussian Process regression and forecasting stock trends

The aim of this project was to learn the mathematical concepts of Gaussian Processes and implement them later on in real-world problems - in adjusted closing prices prediction consisted of three selected stock prices. 

It is obvious that the method developped during this process of creation is not ideal, if it were so I wouldn't share this publictly and made profits myself instead. ;) But nevertheless it can give some good forecasts and be used as another indicator in technical analysis of stock prices as we will see later on below.

## Gaussian Processes

Gaussian processes (Rasmussen and Williams 2006) are a general and flexible class of models for nonlinear regression and classification. They have received attention in the machine learning community over last years, having originally been introduced in geostatistics. They differ from neural networks in that they engage in a full Bayesian treatment, supplying a complete posterior distribution of forecasts. For regression, they are also computationally relatively simple to implement, the basic model requiring only solving a system of linear equations with computational complexity $O(n^3)$.

This section will briefly review Gaussian processes at a level sufficient for understanding the forecasting methodology developed in this project.

## Basic Concepts

A Gaussian process is a generalization of the Gaussian distribution - it represents a probability distribution over *functions* which is entirely specified by a mean and covariance *functions*. Mathematical definition would be then as follows (Rasmussen and Williams 2006):

**Definition:** *A Gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution.*

Let $x$ be some process $f(x)$. We write:
$$f(x)\sim GP(m(\cdot), k(\cdot, \cdot)),$$
where $m(\cdot)$ and $k(\cdot,\cdot)$ are the mean and covariance functions, respectively:
\begin{align*}
m(x)&=E[f(x)] \\
k(x_1, x_2)&=E[(f(x_1)-m(x_1))(f(x_2)-m(x_2))].
\end{align*}
We will assume that we have a training set $D=\lbrace (x_i, y_i) | i=1,\ldots,N \rbrace,$ where $x_i \in \mathbb{R}^D$ and $y_i \in \mathbb{R}$. For sake of simplicity let $X$ be the matrix of all inputs, and $y$ the vector of targets. Also we should assume that the observations $y_i$ from the proces $f(x)$ are noisy:
$$y_i = f(x_i) + \varepsilon_i, \hspace{10mm} \textnormal{where}\hspace{10mm} \varepsilon_i \sim N(0, \sigma_n^2).$$

Regression with a GP is achieved by means of Bayesian inference in order to obtain a posterior distribution over functions given a suitable prior and training data. Then, given new test inputs, we can use the posterior to arrive at a predictive distribution conditional on the test inputs and the training data. 

It is often convienient to assume that the GP prior distribution has mean of zero

$$f(x) \sim GP(0, k(\cdot, \cdot)).$$

Let $f=[f(x_1),\ldots,f(x_n)]$ be a vector of function values in the training set $D$. Their prior distribution is then:
$$f \sim GP(0, K(X, X)),$$
where $K(X,X)_{ij}=k(x_i,x_j)$ is a covariance matrix evaluated using covariance function between given points (also known as *kernel* or *Gram* matrix). Considering the joint prior distribution between training and the test points, with locations given by matrix $X_*$ and whose function values are $f_*$ we can obtain that

$$\begin{bmatrix}
    f       \\
    f_*        \\
\end{bmatrix}\sim
N \left(0,
\begin{bmatrix}
    K(X,X)       & K(X,X_*)  \\
    K(X_*,X)       & K(X_*,X_*) \\
\end{bmatrix} \right).
$$

Then, using Bayes' theorem the join posterior given training data is

$$P(f, f_* \vert y)=\frac{P(y|f,f_*)P(f,f_*)}{P(y)}=\frac{P(y|f)P(f,f_*)}{P(y)},$$

where $P(y\vert f,f_*)=P(y \vert f)$ since the likelihood is conditionally independent of $f_*$ given $f$, and

$$y\vert f \sim N(f, \sigma_n^2 I_n),$$

where $I_n$ is $N\times N$ identity matrix. So the desired predictive distribution is

$$P(f_* \vert y) = \int P(f, f_* \vert y) df=\frac{1}{P(y)}\int P(y \vert f)P(f, f_*)df.$$

Since these distributions are normal, the result of the marginal is also normal we have

\begin{align*}
E[f_*|y]&=K(X_*,X)\Lambda^{-1}y, \\
Cov[f_*|y]&=K(X_*,X_*)-K(X_*,X)\Lambda^{-1}K(X,X_*),
\end{align*}

where $y$ are test points and
$$\Lambda = K(X,X)+\sigma_{n}^{2}I_N.$$

The computation of $\Lambda^{-1}$ is the most computationally expensive in GP regression, requiring as mentioned earlier $O(N^3)$ time and also $O(N^2)$ space.

## Covariance function

A proper choice for the covariance function is important for encoding knowledge about our problem - several examples are given in Rasmussen and Williams (2006). In order to get valid covariance matrices, the covariance function should be symmetric and positive semi-definite, which implies that the all its eigenvalues are positive,

$$ \int k(u,v)f(u)f(v)d \mu(u)d \mu(v) \geq 0,$$
for all functions $f$ defined on appropriate space and measure $\mu$.

The two most common choices for covariance functions are the *squared exponential* (also known as the *Gaussian* or *radial basis function* kernel)

$$k_{SE} (u,v,\sigma_l, \mu)=\mu\exp\left(-\frac{||u-v||^2}{2\sigma_l^2} \right),$$

which we will use in our problem and the *rational quadratic*

$$k_{RQ}(u,v,\sigma_l,\alpha, \mu)=\mu\left( 1+\frac{||u-v||^2}{2\alpha\sigma_l^2} \right)^{-\alpha}.$$

In both cases, the hyperparameter $\sigma_l$ governs the *characteristic length scale* of covariance function, indicating the degree of smoothness of underlying random functions and $\mu$ we can interpret as scaling hyperparameter. The rational quadratic can be interpreted as an infinite mixture of squared exponentials with different length-scales - it converges to a squared exponential with characteristic length-scale $\sigma_l$ as $\alpha \rightarrow \infty$. In this project has been used classical Gasussian kernel

## Hyperparameters optimization

For many machine learning algorithms, this problem has often been approached by minimizing a validation error through croos-validation, but in this case we will apply alternative approach, quite efficient for GP - maximizing the *marginal likelihood* of the observerd data with respect to the hyperparameters. This function can be computed by introducing latent function values that will be integrated over. Let $\theta$ be the set of hyperparameters that have to be optimized, and $K_X(\theta)$ be the covariance matrix computed by given covariance function whore hyperparameters are $\theta$,

$$K_X(\theta)_{i,j}=k(x_i,x_j;\theta).$$

The marginal likelihood then can be wrriten as

$$p(y \vert X, \theta) = \int P(y \vert f, X)p(f \vert X, \theta)df,$$

where the distribution of observations $P(y \vert f, X)$ is conditionally independent of the hyperparameters given the latent function $f$. Under the Gaussian process prior, we have that $f \vert X, \theta \sim N(0, K_X(\theta))$, or in terms of log-likelihood

$$\log P(f | X, \theta) = -\frac{1}{2}f'K_X^{-1}(\theta)-\frac{1}{2}\log |K_X(\theta)|-\frac{N}{2}\log 2\pi.$$

Since the distributions are normal, the marginalization can be done analytically to yield

$$\log P(y | X, \theta) = -\frac{1}{2}y'(K_X(\theta)+\sigma^2_n I_N)^{-1}y-\frac{1}{2}\log |K_X(\theta)+\sigma^2_n I_N|-\frac{N}{2}\log 2\pi.$$

This expression can be maximized numerically, for instance by a conjugate gradient or like in our case - the default python's sklearn optimizer ```fmin_l_bfgs_b``` to yield the selected hyperparameters:

$$\theta^{*}=\arg\max_\theta\log p(y \vert X, \theta).$$

The gradient of the marginal log-likelihood with respect to the hyperparameters — necessary for numerical optimization algorithms — can be expressed as

$$\frac{\partial \log P(f | X, \theta)}{\partial \theta_i} = -\frac{1}{2}y'K_X^{-1}(\theta)\frac{\partial K_X(\theta)}{\partial \theta_i}K_X^{-1}(\theta)y-\frac{1}{2}Tr\left( K_X^{-1}(\theta)\frac{\partial K_X(\theta)}{\partial \theta_i} \right).$$

See Rasmussen and Williams (2006) for details on the derivation of this equation.

## Forecasting Methodology

The main idea of this approach is to avoid representing the whole history as one time series. Each time series is treated as an independent input variable in the regression model. Consider a set of $N$ real time series each of length $M_i$, $\lbrace y^i_t \rbrace$, $$i=1,\ldots,N$$ and $t=1,\ldots,M_i$. In this application each $i$ represents a different year, and the series is the sequence of a particular prices during the period where it is traded. Considering the length of the stock market year, usually $M$ will be equal to $252$ and sometimes less if incomplete series is considered (for example this year) assuming that the series follow an annual cycle. Thus knowledge from past series can be transferred to a new one to be forecast. Each trade year of data is treated as a separate time series and the corresponding year is used as a independent variable in regression model.

The forecasting problem is that given observations from the complete series $i=1,\ldots,N-1$ and (optionally) from a partial last series $\lbrace y_t^N \rbrace$, $t=1,\ldots,M_N$, we want to extrapolate the last series until predetermined endpoint (usually a multiple of a quarter during a year) - characterize the joint distribution of $\lbrace y_\tau^N \rbrace$, $\tau=M_N+1,\ldots,M_N+H$ for some $H$. We are also given a set of non-stochastic explanatory variables specific to each series, $\lbrace x_t^i \rbrace$, where $x_t^i \in \mathbb{R}^d$. Our objective is to find an effective representation of $P(\lbrace y_{\tau}^N \rbrace_{\tau=M_N+1,\ldots,M_N+H} \vert \lbrace x_t^i, y_t^i \rbrace_{t=1,\ldots,M_i}^{i=1,\ldots,N} )$, with $\tau, i$ and $t$ ranging, respectively over the forecatsing horizon, the available series and the observations within a series.

Everything mentioned in this section was implemented in Python using the wonderful library ```sklearn```, mainly the ```sklearn.gaussian_process```.

## Selected data and Evaluation

For this project, three stocks/indices were selected:
* S&P 500 (GSPC),
* Treasury Yield 5 Years (FVX),
* Starbucks (SBUX).

The daily changes of adjusted closing prices of these stocks were examined and the historical data was downloaded from the yahoo finance section.

## S&P 500 (GSPC)

## Treasury Yield 5 Years (FVX)

## Starbucks (SBUX)

## Summary

## Bibliography
* Bengio, Chapados, Forecasting and Trading Commodity Contract Spreads with Gaussian Processes, 2007,
* Correa, Farrell, Gaussian Process Regression Models for Predicting Stock Trends, 2007,
* Chen, Gaussian process regression methods and extensions for stock market prediction, 2017
* Rasmussen, Williams, Gaussian Processes for Machine Learning, 2006