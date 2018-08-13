# Gaussian Process regression and forecasting stock trends

The aim of this project was to learn the mathematical concepts of Gaussian Processes and implement them later on in real-world problems - in adjusted closing prices prediction consisted of three selected stock prices. 

It is obvious that the method developped during this process of creation is not ideal, if it were so I wouldn't share this publictly and made profits myself instead. ;) But nevertheless it can give some good forecasts and be used as another indicator in technical analysis of stock prices as we will see later on below.

## Gaussian Processes

Gaussian processes (Rasmussen and Williams 2006) are a general and flexible class of models for nonlinear regression and classification. They have received attention in the machine learning community over last years, having originally been introduced in geostatistics. They differ from neural networks in that they engage in a full Bayesian treatment, supplying a complete posterior distribution of forecasts. For regression, they are also computationally relatively simple to implement, the basic model requiring only solving a system of linear equations with computational complexity <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/90846c243bb784093adbb6d2d0b2b9d0.svg?invert_in_darkmode" align=middle width=43.022265pt height=26.76201000000001pt/>.

This section will briefly review Gaussian processes at a level sufficient for understanding the forecasting methodology developed in this project.

## Basic Concepts

A Gaussian process is a generalization of the Gaussian distribution - it represents a probability distribution over *functions* which is entirely specified by a mean and covariance *functions*. Mathematical definition would be then as follows (Rasmussen and Williams 2006):

**Definition:** *A Gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution.*

Let <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.395100000000005pt height=14.155350000000013pt/> be some process <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode" align=middle width=31.997955pt height=24.65759999999998pt/>. We write:
<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/61165782247ba54364ead2d26eb29801.svg?invert_in_darkmode" align=middle width=174.41819999999998pt height=16.438356pt/></p>
where <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/5c08e143f5965ada34879fd8fca7f2ec.svg?invert_in_darkmode" align=middle width=31.784775000000003pt height=24.65759999999998pt/> and <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/9002285f1fadb93025a8f8e11caf49b3.svg?invert_in_darkmode" align=middle width=38.29914000000001pt height=24.65759999999998pt/> are the mean and covariance functions, respectively:
<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/f3450affc2d4715c5e53e8f2faa0c0a8.svg?invert_in_darkmode" align=middle width=343.87815pt height=41.09589pt/></p>
We will assume that we have a training set <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/babb947beaeeb67b62cf30dde7c1a4e1.svg?invert_in_darkmode" align=middle width=197.37580499999999pt height=24.65759999999998pt/> where <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/56d9186dece2f947a7466bef868b3519.svg?invert_in_darkmode" align=middle width=57.933315pt height=27.656969999999987pt/> and <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/30c9b0271969f8469d8017226c2d033b.svg?invert_in_darkmode" align=middle width=45.495615pt height=22.64855999999997pt/>. For sake of simplicity let <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908740000000003pt height=22.46574pt/> be the matrix of all inputs, and <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649300000000004pt height=14.155350000000013pt/> the vector of targets. Also we should assume that the observations <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710445000000004pt height=14.155350000000013pt/> from the proces <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode" align=middle width=31.997955pt height=24.65759999999998pt/> are noisy:
<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/c9d5e04d1f686bd01e9d318cf07dd84c.svg?invert_in_darkmode" align=middle width=350.328pt height=18.312359999999998pt/></p>

Regression with a GP is achieved by means of Bayesian inference in order to obtain a posterior distribution over functions given a suitable prior and training data. Then, given new test inputs, we can use the posterior to arrive at a predictive distribution conditional on the test inputs and the training data. 

It is often convienient to assume that the GP prior distribution has mean of zero

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/366206f9d431d65fa88bdd48f9757d50.svg?invert_in_darkmode" align=middle width=150.852735pt height=16.438356pt/></p>

Let <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/8a3fc29a7682963cae01d077bdf09932.svg?invert_in_darkmode" align=middle width=157.714755pt height=24.65759999999998pt/> be a vector of function values in the training set <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode" align=middle width=14.066250000000002pt height=22.46574pt/>. Their prior distribution is then:
<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/ef8f3c6bce47e090252f0a1a30fff555.svg?invert_in_darkmode" align=middle width=154.50566999999998pt height=16.438356pt/></p>
where <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/e77f55d98299148e2d2a5c5c3168df02.svg?invert_in_darkmode" align=middle width=157.98370500000001pt height=24.65759999999998pt/> is a covariance matrix evaluated using covariance function between given points (also known as *kernel* or *Gram* matrix). Considering the joint prior distribution between training and the test points, with locations given by matrix <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/45ec301c305b1c922524016ea2179f55.svg?invert_in_darkmode" align=middle width=20.354070000000004pt height=22.46574pt/> and whose function values are <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/4d00376b927ae39d0a206dc721cfc59f.svg?invert_in_darkmode" align=middle width=14.783175000000002pt height=22.831379999999992pt/> we can obtain that

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/10d98a532b02f33304b2328c70806732.svg?invert_in_darkmode" align=middle width=302.32949999999994pt height=39.45249pt/></p>

Then, using Bayes' theorem the join posterior given training data is

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/b15b4f2176f1fd36ff3736fd7f6ab106.svg?invert_in_darkmode" align=middle width=361.12725pt height=38.834894999999996pt/></p>

where <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/ce239ce8e1bcbb965c18a0645d1ac566.svg?invert_in_darkmode" align=middle width=141.22548pt height=24.65759999999998pt/> since the likelihood is conditionally independent of <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/4d00376b927ae39d0a206dc721cfc59f.svg?invert_in_darkmode" align=middle width=14.783175000000002pt height=22.831379999999992pt/> given <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.817500000000004pt height=22.831379999999992pt/>, and

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/2a60134d467edebd40789ae2ae76e6a3.svg?invert_in_darkmode" align=middle width=128.02713pt height=18.312359999999998pt/></p>

where <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/04c9bb763257fc1746a9005d84484716.svg?invert_in_darkmode" align=middle width=15.352095000000004pt height=22.46574pt/> is <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/0ef69de18444d6cd8f1e8e13faf27443.svg?invert_in_darkmode" align=middle width=50.091195pt height=22.46574pt/> identity matrix. So the desired predictive distribution is

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/8e87b85159447a0186cf71d75407cdda.svg?invert_in_darkmode" align=middle width=389.25645pt height=37.760085pt/></p>

Since these distributions are normal, the result of the marginal is also normal we have

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/600c4c216b73944c6b2a62927ef38bba.svg?invert_in_darkmode" align=middle width=362.33504999999997pt height=45.00837pt/></p>

where <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649300000000004pt height=14.155350000000013pt/> are test points and
<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/e759fd90c1b12009d40e59ef742f8fd0.svg?invert_in_darkmode" align=middle width=160.158075pt height=18.312359999999998pt/></p>

The computation of <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/84dc4a86e5fd3dce29cfe83d6241e14a.svg?invert_in_darkmode" align=middle width=28.242225pt height=26.76201000000001pt/> is the most computationally expensive in GP regression, requiring as mentioned earlier <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/0e10b94fd93211f59f66cd90dec1abe1.svg?invert_in_darkmode" align=middle width=48.155415000000005pt height=26.76201000000001pt/> time and also <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/c3f65f86f2baa7f28840d7c68c00f5f2.svg?invert_in_darkmode" align=middle width=48.155415000000005pt height=26.76201000000001pt/> space.

## Covariance function

A proper choice for the covariance function is important for encoding knowledge about our problem - several examples are given in Rasmussen and Williams (2006). In order to get valid covariance matrices, the covariance function should be symmetric and positive semi-definite, which implies that the all its eigenvalues are positive,

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/54436ad02fecb2d0a14aa7a1aa184959.svg?invert_in_darkmode" align=middle width=244.65044999999998pt height=36.53001pt/></p>
for all functions <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.817500000000004pt height=22.831379999999992pt/> defined on appropriate space and measure <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.904950000000003pt height=14.155350000000013pt/>.

The two most common choices for covariance functions are the *squared exponential* (also known as the *Gaussian* or *radial basis function* kernel)

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/55ba6349062f125c5dc6f60c12b1b634.svg?invert_in_darkmode" align=middle width=279.73109999999997pt height=40.73619pt/></p>

which we will use in our problem and the *rational quadratic*

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/88f0f5d2593424176ac9fe5b4b1758d8.svg?invert_in_darkmode" align=middle width=306.2565pt height=43.79562pt/></p>

In both cases, the hyperparameter <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/f36211b3b322fc8dd3c8c1d2712cc8c5.svg?invert_in_darkmode" align=middle width=13.616955000000004pt height=14.155350000000013pt/> governs the *characteristic length scale* of covariance function, indicating the degree of smoothness of underlying random functions and <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.904950000000003pt height=14.155350000000013pt/> we can interpret as scaling hyperparameter. The rational quadratic can be interpreted as an infinite mixture of squared exponentials with different length-scales - it converges to a squared exponential with characteristic length-scale <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/f36211b3b322fc8dd3c8c1d2712cc8c5.svg?invert_in_darkmode" align=middle width=13.616955000000004pt height=14.155350000000013pt/> as <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/35e046fde190057d22735c350f34d82a.svg?invert_in_darkmode" align=middle width=52.5855pt height=14.155350000000013pt/>. In this project has been used classical Gasussian kernel

## Hyperparameters optimization

For many machine learning algorithms, this problem has often been approached by minimizing a validation error through croos-validation, but in this case we will apply alternative approach, quite efficient for GP - maximizing the *marginal likelihood* of the observerd data with respect to the hyperparameters. This function can be computed by introducing latent function values that will be integrated over. Let <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.173588500000005pt height=22.831379999999992pt/> be the set of hyperparameters that have to be optimized, and <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/94e240a6c38dd57559862f348c9ed09c.svg?invert_in_darkmode" align=middle width=47.416875000000005pt height=24.65759999999998pt/> be the covariance matrix computed by given covariance function whore hyperparameters are <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.173588500000005pt height=22.831379999999992pt/>,

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/d8c56b3fafd7f6b5e82127e8c6706f9a.svg?invert_in_darkmode" align=middle width=165.2178pt height=17.031959999999998pt/></p>

The marginal likelihood then can be wrriten as

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/ba67899f929339faa716bf5562172dd0.svg?invert_in_darkmode" align=middle width=258.99885pt height=36.53001pt/></p>

where the distribution of observations <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/830cddb3bf621c2d41ad68bd90fbcf7c.svg?invert_in_darkmode" align=middle width=69.95637pt height=24.65759999999998pt/> is conditionally independent of the hyperparameters given the latent function <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.817500000000004pt height=22.831379999999992pt/>. Under the Gaussian process prior, we have that <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/d43fd41af54de194cb575dbec817d9d1.svg?invert_in_darkmode" align=middle width=156.50349000000003pt height=24.65759999999998pt/>, or in terms of log-likelihood

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/34691d54fc242c644ceea4e34f1cb3b3.svg?invert_in_darkmode" align=middle width=412.50494999999995pt height=33.629475pt/></p>

Since the distributions are normal, the marginalization can be done analytically to yield

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/e93d33a1101341daf1a444f91109ccbf.svg?invert_in_darkmode" align=middle width=559.17675pt height=33.629475pt/></p>

This expression can be maximized numerically, for instance by a conjugate gradient or like in our case - the default python's sklearn optimizer ```fmin_l_bfgs_b``` to yield the selected hyperparameters:

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/835c503b35ae397cf458702511257dcd.svg?invert_in_darkmode" align=middle width=189.11145pt height=23.059245pt/></p>

The gradient of the marginal log-likelihood with respect to the hyperparameters — necessary for numerical optimization algorithms — can be expressed as

<p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/641e04f68383e56782780bf32efef434.svg?invert_in_darkmode" align=middle width=546.08565pt height=39.45249pt/></p>

See Rasmussen and Williams (2006) for details on the derivation of this equation.

## Forecasting Methodology

The main idea of this approach is to avoid representing the whole history as one time series. Each time series is treated as an independent input variable in the regression model. Consider a set of <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.999985000000004pt height=22.46574pt/> real time series each of length <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/e8a87898efc00bd6e44ae2c7edcfcd1c.svg?invert_in_darkmode" align=middle width=20.598435000000002pt height=22.46574pt/>, <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/03cb7eaae2ee9d88d11183776ebd4ae4.svg?invert_in_darkmode" align=middle width=30.560475000000004pt height=27.159000000000013pt/>, <p align="center"><img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/102fda38deabc3fb86ef544c4e67e469.svg?invert_in_darkmode" align=middle width=87.32955pt height=14.429217pt/></p> and <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/a794aecd97251ba516979054e5f96bb6.svg?invert_in_darkmode" align=middle width=93.20091000000001pt height=22.46574pt/>. In this application each <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663295000000005pt height=21.683310000000006pt/> represents a different year, and the series is the sequence of a particular prices during the period where it is traded. Considering the length of the stock market year, usually <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode" align=middle width=17.739810000000002pt height=22.46574pt/> will be equal to <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/c8d6a3a03a14a5da0ba54d187dcad53a.svg?invert_in_darkmode" align=middle width=24.657765pt height=21.18732pt/> and sometimes less if incomplete series is considered (for example this year) assuming that the series follow an annual cycle. Thus knowledge from past series can be transferred to a new one to be forecast. Each trade year of data is treated as a separate time series and the corresponding year is used as a independent variable in regression model.

The forecasting problem is that given observations from the complete series <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/34f8cf8effcc1c7088892ebc5a56d8f3.svg?invert_in_darkmode" align=middle width=115.63991999999999pt height=22.46574pt/> and (optionally) from a partial last series <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/e39486f57b128be8c93e139c296b080a.svg?invert_in_darkmode" align=middle width=37.555815pt height=27.656969999999987pt/>, <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/9711afd060107b551840bca98443b740.svg?invert_in_darkmode" align=middle width=100.196085pt height=22.46574pt/>, we want to extrapolate the last series until predetermined endpoint (usually a multiple of a quarter during a year) - characterize the joint distribution of <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/053befb5718e8998da035e8a1228dc94.svg?invert_in_darkmode" align=middle width=37.555815pt height=27.656969999999987pt/>, <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/fc12f6a9bf12c8b4a0afba93319dcf61.svg?invert_in_darkmode" align=middle width=187.72660499999998pt height=22.46574pt/> for some <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.999985000000004pt height=22.46574pt/>. We are also given a set of non-stochastic explanatory variables specific to each series, <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/d44c3b9e9c7bc63c4f5887074ee41750.svg?invert_in_darkmode" align=middle width=31.621095000000004pt height=27.159000000000013pt/>, where <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/8be904a9f5e36cf5797a798430597e46.svg?invert_in_darkmode" align=middle width=53.989155pt height=27.91271999999999pt/>. Our objective is to find an effective representation of <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/424be0cf8e8ccf88eda96d764bab8cc1.svg?invert_in_darkmode" align=middle width=302.88505499999997pt height=31.52523pt/>, with <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/2048a9cff1e3d55aef60e1f9c96002eb.svg?invert_in_darkmode" align=middle width=21.102840000000004pt height=21.683310000000006pt/> and <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936155500000004pt height=20.222069999999988pt/> ranging, respectively over the forecatsing horizon, the available series and the observations within a series.

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