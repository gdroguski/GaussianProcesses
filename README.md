# Gaussian Process Regression and Forecasting Stock Trends

<p>The aim of this project was to learn the mathematical concepts of Gaussian Processes and implement them later on in real-world problems - in adjusted closing price trend prediction consisted of three selected stock entities.</p>

<p>It is obvious that the method developped during this process of creation is not ideal, if it were so I wouldn't share this publictly and made profits myself instead. ;) But nevertheless it can give some good forecasts and be used as another indicator in technical analysis of stock prices as we will see later on below.</p>

The project is written in ```Python 3``` and to run it you will need also ```matplotlib```, ```numpy```, ```pandas``` and ```sklearn``` libraries. As written in [Data and Evaluation](#data-and-evaluation) section, the data was downloaded from yahoo finance section in the ```csv``` format so basically if you'd like to test this model on some other companies from yahoo you only have to download and put it into ```Data``` directory within the project. 

Worth mentioning is the fact that to run the analysis you have to run the ```data_presenter.py``` file and the project will do it's magic for every file in ```Data``` directory. So to run just type ```python data_presenter.py```.

## Table of Contents
* [Gaussian Processes](#gaussian-processes)
  * Basic Concepts
  * Covariance Function
  * Hyperparameters Optimization
* [Forecasting Methodology](#forecasting-methodology)
* [Data and Evaluation](#data-and-evaluation)
  * S&P 500 (GSPC)
  * The Boeing Company (BA)
  * Starbucks (SBUX)
* [Summary](#summary)
* [Bibliography](#bibliography)


## Gaussian Processes


<p>Gaussian processes are a general and flexible class of models for nonlinear regression and classification. They have received attention in the machine learning community over last years, having originally been introduced in geostatistics. They differ from neural networks in that they engage in a full Bayesian treatment, supplying a complete posterior distribution of forecasts. For regression, they are also computationally relatively simple to implement, the basic model requiring only solving a system of linear equations with computational complexity <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/90846c243bb784093adbb6d2d0b2b9d0.svg" align=middle width=43.022265pt height=26.76201000000001pt/>.</p>

<p>This section will briefly review Gaussian processes at a level sufficient for understanding the forecasting methodology developed in this project.</p>

## Basic Concepts

<p>A Gaussian process is a generalization of the Gaussian distribution - it represents a probability distribution over <i>functions</i> which is entirely specified by a mean and covariance <i>functions</i>. Mathematical definition would be then as follows:</p>

**Definition:** *A Gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution.*

<p>Let <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/332cc365a4987aacce0ead01b8bdcc0b.svg" align=middle width=9.395100000000005pt height=14.155350000000013pt/> be some process <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/7997339883ac20f551e7f35efff0a2b9.svg" align=middle width=31.997955pt height=24.65759999999998pt/>. We write:</p>
<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/61165782247ba54364ead2d26eb29801.svg" align=middle width=174.41819999999998pt height=16.438356pt/></p>
<p>where <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/5c08e143f5965ada34879fd8fca7f2ec.svg" align=middle width=31.784775000000003pt height=24.65759999999998pt/> and <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/9002285f1fadb93025a8f8e11caf49b3.svg" align=middle width=38.29914000000001pt height=24.65759999999998pt/> are the mean and covariance functions, respectively:</p>
<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/f3450affc2d4715c5e53e8f2faa0c0a8.svg" align=middle width=343.87815pt height=41.09589pt/></p>
<p>We will assume that we have a training set <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/babb947beaeeb67b62cf30dde7c1a4e1.svg" align=middle width=197.37580499999999pt height=24.65759999999998pt/> where <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/56d9186dece2f947a7466bef868b3519.svg" align=middle width=57.933315pt height=27.656969999999987pt/> and <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/30c9b0271969f8469d8017226c2d033b.svg" align=middle width=45.495615pt height=22.64855999999997pt/>. For sake of simplicity let <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/cbfb1b2a33b28eab8a3e59464768e810.svg" align=middle width=14.908740000000003pt height=22.46574pt/> be the matrix of all inputs, and <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/deceeaf6940a8c7a5a02373728002b0f.svg" align=middle width=8.649300000000004pt height=14.155350000000013pt/> the vector of targets. Also we should assume that the observations <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/2b442e3e088d1b744730822d18e7aa21.svg" align=middle width=12.710445000000004pt height=14.155350000000013pt/> from the proces <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/7997339883ac20f551e7f35efff0a2b9.svg" align=middle width=31.997955pt height=24.65759999999998pt/> are noisy:</p>
<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/c9d5e04d1f686bd01e9d318cf07dd84c.svg" align=middle width=350.328pt height=18.312359999999998pt/></p>

<p>Regression with a GP is achieved by means of Bayesian inference in order to obtain a posterior distribution over functions given a suitable prior and training data. Then, given new test inputs, we can use the posterior to arrive at a predictive distribution conditional on the test inputs and the training data. </p>

<p>It is often convienient to assume that the GP prior distribution has mean of zero</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/366206f9d431d65fa88bdd48f9757d50.svg" align=middle width=150.852735pt height=16.438356pt/></p>

<p>Let <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/8a3fc29a7682963cae01d077bdf09932.svg" align=middle width=157.714755pt height=24.65759999999998pt/> be a vector of function values in the training set <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/78ec2b7008296ce0561cf83393cb746d.svg" align=middle width=14.066250000000002pt height=22.46574pt/>. Their prior distribution is then:</p>
<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/ef8f3c6bce47e090252f0a1a30fff555.svg" align=middle width=154.50566999999998pt height=16.438356pt/></p>
<p>where <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/e77f55d98299148e2d2a5c5c3168df02.svg" align=middle width=157.98370500000001pt height=24.65759999999998pt/> is a covariance matrix evaluated using covariance function between given points (also known as <i>kernel</i> or <i>Gram</i> matrix). Considering the joint prior distribution between training and the test points, with locations given by matrix <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/45ec301c305b1c922524016ea2179f55.svg" align=middle width=20.354070000000004pt height=22.46574pt/> and whose function values are <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/4d00376b927ae39d0a206dc721cfc59f.svg" align=middle width=14.783175000000002pt height=22.831379999999992pt/> we can obtain that</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/10d98a532b02f33304b2328c70806732.svg" align=middle width=302.32949999999994pt height=39.45249pt/></p>

<p>Then, using Bayes' theorem the join posterior given training data is</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/b15b4f2176f1fd36ff3736fd7f6ab106.svg" align=middle width=361.12725pt height=38.834894999999996pt/></p>

<p>where <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/ce239ce8e1bcbb965c18a0645d1ac566.svg" align=middle width=141.22548pt height=24.65759999999998pt/> since the likelihood is conditionally independent of <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/4d00376b927ae39d0a206dc721cfc59f.svg" align=middle width=14.783175000000002pt height=22.831379999999992pt/> given <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/190083ef7a1625fbc75f243cffb9c96d.svg" align=middle width=9.817500000000004pt height=22.831379999999992pt/>, and</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/2a60134d467edebd40789ae2ae76e6a3.svg" align=middle width=128.02713pt height=18.312359999999998pt/></p>

<p>where <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/04c9bb763257fc1746a9005d84484716.svg" align=middle width=15.352095000000004pt height=22.46574pt/> is <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/0ef69de18444d6cd8f1e8e13faf27443.svg" align=middle width=50.091195pt height=22.46574pt/> identity matrix. So the desired predictive distribution is</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/8e87b85159447a0186cf71d75407cdda.svg" align=middle width=389.25645pt height=37.760085pt/></p>

<p>Since these distributions are normal, the result of the marginal is also normal we have</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/600c4c216b73944c6b2a62927ef38bba.svg" align=middle width=362.33504999999997pt height=45.00837pt/></p>

<p>where <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/deceeaf6940a8c7a5a02373728002b0f.svg" align=middle width=8.649300000000004pt height=14.155350000000013pt/> are test points and</p>
<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/e759fd90c1b12009d40e59ef742f8fd0.svg" align=middle width=160.158075pt height=18.312359999999998pt/></p>

<p>The computation of <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/84dc4a86e5fd3dce29cfe83d6241e14a.svg" align=middle width=28.242225pt height=26.76201000000001pt/> is the most computationally expensive in GP regression, requiring as mentioned earlier <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/0e10b94fd93211f59f66cd90dec1abe1.svg" align=middle width=48.155415000000005pt height=26.76201000000001pt/> time and also <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/c3f65f86f2baa7f28840d7c68c00f5f2.svg" align=middle width=48.155415000000005pt height=26.76201000000001pt/> space.</p>

## Covariance Function

<p>A proper choice for the covariance function is important for encoding knowledge about our problem - several examples are given in Rasmussen and Williams (2006). In order to get valid covariance matrices, the covariance function should be symmetric and positive semi-definite, which implies that the all its eigenvalues are positive,</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/54436ad02fecb2d0a14aa7a1aa184959.svg" align=middle width=244.65044999999998pt height=36.53001pt/></p>
<p>for all functions <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/190083ef7a1625fbc75f243cffb9c96d.svg" align=middle width=9.817500000000004pt height=22.831379999999992pt/> defined on appropriate space and measure <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/07617f9d8fe48b4a7b3f523d6730eef0.svg" align=middle width=9.904950000000003pt height=14.155350000000013pt/>.</p>

<p>The two most common choices for covariance functions are the <i>squared exponential</i> (also known as the <i>Gaussian</i> or <i>radial basis function</i> kernel):</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/55ba6349062f125c5dc6f60c12b1b634.svg" align=middle width=279.73109999999997pt height=40.73619pt/></p>

<p>which we will use in our problem and the <i>rational quadratic</i>:</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/88f0f5d2593424176ac9fe5b4b1758d8.svg" align=middle width=306.2565pt height=43.79562pt/></p>

<p>In both cases, the hyperparameter <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/f36211b3b322fc8dd3c8c1d2712cc8c5.svg" align=middle width=13.616955000000004pt height=14.155350000000013pt/> governs the <i>characteristic length scale</i> of covariance function, indicating the degree of smoothness of underlying random functions and <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/07617f9d8fe48b4a7b3f523d6730eef0.svg" align=middle width=9.904950000000003pt height=14.155350000000013pt/> we can interpret as scaling hyperparameter. The rational quadratic can be interpreted as an infinite mixture of squared exponentials with different length-scales - it converges to a squared exponential with characteristic length-scale <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/f36211b3b322fc8dd3c8c1d2712cc8c5.svg" align=middle width=13.616955000000004pt height=14.155350000000013pt/> as <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/35e046fde190057d22735c350f34d82a.svg" align=middle width=52.5855pt height=14.155350000000013pt/>. In this project has been used classical Gasussian kernel.</p>

## Hyperparameters Optimization

<p>For many machine learning algorithms, this problem has often been approached by minimizing a validation error through cross-validation, but in this case we will apply alternative approach, quite efficient for GP - maximizing the <i>marginal likelihood</i> of the observerd data with respect to the hyperparameters. This function can be computed by introducing latent function values that will be integrated over. Let <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/27e556cf3caa0673ac49a8f0de3c73ca.svg" align=middle width=8.173588500000005pt height=22.831379999999992pt/> be the set of hyperparameters that have to be optimized, and <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/94e240a6c38dd57559862f348c9ed09c.svg" align=middle width=47.416875000000005pt height=24.65759999999998pt/> be the covariance matrix computed by given covariance function whose hyperparameters are <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/27e556cf3caa0673ac49a8f0de3c73ca.svg" align=middle width=8.173588500000005pt height=22.831379999999992pt/>,</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/d8c56b3fafd7f6b5e82127e8c6706f9a.svg" align=middle width=165.2178pt height=17.031959999999998pt/></p>

<p>The marginal likelihood then can be wrriten as</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/ba67899f929339faa716bf5562172dd0.svg" align=middle width=258.99885pt height=36.53001pt/></p>

<p>where the distribution of observations <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/830cddb3bf621c2d41ad68bd90fbcf7c.svg" align=middle width=69.95637pt height=24.65759999999998pt/> is conditionally independent of the hyperparameters given the latent function <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/190083ef7a1625fbc75f243cffb9c96d.svg" align=middle width=9.817500000000004pt height=22.831379999999992pt/>. Under the Gaussian process prior, we have that <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/d43fd41af54de194cb575dbec817d9d1.svg" align=middle width=156.50349000000003pt height=24.65759999999998pt/>, or in terms of log-likelihood</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/34691d54fc242c644ceea4e34f1cb3b3.svg" align=middle width=412.50494999999995pt height=33.629475pt/></p>

<p>Since the distributions are normal, the marginalization can be done analytically to yield</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/e93d33a1101341daf1a444f91109ccbf.svg" align=middle width=559.17675pt height=33.629475pt/></p>

This expression can be maximized numerically, for instance by a conjugate gradient or like in our case - the default python's sklearn optimizer ```fmin_l_bfgs_b``` to yield the selected hyperparameters:

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/835c503b35ae397cf458702511257dcd.svg" align=middle width=189.11145pt height=23.059245pt/></p>

<p>The gradient of the marginal log-likelihood with respect to the hyperparameters-necessary for numerical optimization algorithms—can be expressed as</p>

<p align="center"><img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/641e04f68383e56782780bf32efef434.svg" align=middle width=546.08565pt height=39.45249pt/></p>

<p>See Rasmussen and Williams (2006) for details on the derivation of this equation.</p>

## Forecasting Methodology

<p>The main idea of this approach is to avoid representing the whole history as one time series. Each time series is treated as an independent input variable in the regression model. Consider a set of <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/f9c4988898e7f532b9f826a75014ed3c.svg" align=middle width=14.999985000000004pt height=22.46574pt/> real time series each of length <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/e8a87898efc00bd6e44ae2c7edcfcd1c.svg" align=middle width=20.598435000000002pt height=22.46574pt/>, <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/03cb7eaae2ee9d88d11183776ebd4ae4.svg" align=middle width=30.560475000000004pt height=27.159000000000013pt/>, <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/102fda38deabc3fb86ef544c4e67e469.svg" align=middle width=87.32955pt height=14.429217pt/> and <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/a794aecd97251ba516979054e5f96bb6.svg" align=middle width=93.20091000000001pt height=22.46574pt/>. In this application each <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/77a3b857d53fb44e33b53e4c8b68351a.svg" align=middle width=5.663295000000005pt height=21.683310000000006pt/> represents a different year, and the series is the sequence of a particular prices during the period where it is traded. Considering the length of the stock market year, usually <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/fb97d38bcc19230b0acd442e17db879c.svg" align=middle width=17.739810000000002pt height=22.46574pt/> will be equal to <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/c8d6a3a03a14a5da0ba54d187dcad53a.svg" align=middle width=24.657765pt height=21.18732pt/> and sometimes less if incomplete series is considered (for example this year) assuming that the series follow an annual cycle. Thus knowledge from past series can be transferred to a new one to be forecast. Each trade year of data is treated as a separate time series and the corresponding year is used as a independent variable in regression model.</p>

<p>The forecasting problem is that given observations from the complete series <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/34f8cf8effcc1c7088892ebc5a56d8f3.svg" align=middle width=115.63991999999999pt height=22.46574pt/> and (optionally) from a partial last series <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/e39486f57b128be8c93e139c296b080a.svg" align=middle width=37.555815pt height=27.656969999999987pt/>, <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/9711afd060107b551840bca98443b740.svg" align=middle width=100.196085pt height=22.46574pt/>, we want to extrapolate the last series until predetermined endpoint (usually a multiple of a quarter length during a year) - characterize the joint distribution of <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/053befb5718e8998da035e8a1228dc94.svg" align=middle width=37.555815pt height=27.656969999999987pt/>, <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/fc12f6a9bf12c8b4a0afba93319dcf61.svg" align=middle width=187.72660499999998pt height=22.46574pt/> for some <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/7b9a0316a2fcd7f01cfd556eedf72e96.svg" align=middle width=14.999985000000004pt height=22.46574pt/>. We are also given a set of non-stochastic explanatory variables specific to each series, <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/d44c3b9e9c7bc63c4f5887074ee41750.svg" align=middle width=31.621095000000004pt height=27.159000000000013pt/>, where <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/8be904a9f5e36cf5797a798430597e46.svg" align=middle width=53.989155pt height=27.91271999999999pt/>. Our objective is to find an effective representation of <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/424be0cf8e8ccf88eda96d764bab8cc1.svg" align=middle width=302.88505499999997pt height=31.52523pt/>, with <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/2048a9cff1e3d55aef60e1f9c96002eb.svg" align=middle width=21.102840000000004pt height=21.683310000000006pt/> and <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/4f4f4e395762a3af4575de74c019ebb5.svg" align=middle width=5.936155500000004pt height=20.222069999999988pt/> ranging, respectively over the forecatsing horizon, the available series and the observations within a series.</p>

Everything mentioned in this section was implemented in Python using the wonderful library ```sklearn```, mainly the ```sklearn.gaussian_process```.

## Data and Evaluation

For this project, three stocks/indices were selected:
* S&P 500 (GSPC),
* The Boeing Company (BA),
* Starbucks (SBUX).

<p>The daily changes of adjusted closing prices of these stocks were examined and the historical data was downloaded in the form of csv file from the yahoo finance section. There are two sample periods taken for these three indices: first based on years 2008-2016 for prediction of whole year 2017 and second based on years 2008-2018 (up to end of second quarter) for prediction of the rest of the 2018 year. We have about 252 days of trading year per years since no data is observed on weekends. However, some years have more than 252 days of trading and some less so we choose to ignore 252+ days and for those with less trading days the remaing few we fill up with mean of the year to have equal <img src="https://github.com/gdroguski/GaussianProcesses/blob/master/Pics/Latex/703ec2efcf70940484eac9c4ccde49ec.svg" align=middle width=64.31501999999999pt height=22.46574pt/> for all years. </p>

We choose to use adjusted close prices because we aim to predict the trend of the stocks not the prices. The adjusted close price is used to avoid the effect of dividends and splits because when stock has a split, its price drops by half. The adjusted close prices are standardized to zero mean and unit standard deviation. We also normalize the prices in each year to avoid the variation from previous years by subtracting the first day to start from zero. 

## S&P 500 (GSPC)

We will begin with testing our model on GSPC index. Its corresponding prices chart through history is as follows:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/GSPC/GSPC_whole_prices.png">
</p>

Where the last two vertical lines on the right corresponds to year 2018.

First considered sample period is 2008-2016 so lets take a look at its chart:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/GSPC/GSPC_2008_2016_prices.png">
</p>

Normalized prices for this period will be respectively:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/GSPC/GSPC_2008_2016_normalized_prices.png">
</p>

So our prediction for the whole year 2017 with 95% confidence intervals will be:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/GSPC/GSPC_2017_prediction.png">
</p>

As we can see there are some variations but also the predicted trend for the price is quite good.

Nextly lets consider the 2008-2018 period including first two quarters:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/GSPC/GSPC_2008_2018_prices.png">
</p>

And our prediction for the rest of 2018 will be:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/GSPC/GSPC_2018_prediction.png">
</p>

At the beginning prediction drifts away from the price but we can see further some trend going up so it remains for us to wait to the end of the 2018 and compare our forecast with actual price.

## The Boeing Company (BA)

Now lets test our model on BA price. Its corresponding prices chart through history is as follows:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/BA/BA_whole_prices.png">
</p>

Where the last two vertical lines on the right also corresponds to year 2018.

First considered sample period again is 2008-2016 so lets take a look at its chart:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/BA/BA_2008_2016_prices.png">
</p>

Normalized prices for this period will be respectively:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/BA/BA_2008_2016_normalized_prices.png">
</p>

Our prediction for the whole year 2017 with 95% confidence intervals will be:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/BA/BA_2017_prediction.png">
</p>

This time our model at the beginning of the year underestimates the price behaviour but later on the trend of the forecast is quite well covering the prices trend direction.

Nextly lets consider the 2008-2018 period including first two quarters:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/BA/BA_2008_2018_prices.png">
</p>

And our prediction for the rest of 2018 will be:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/BA/BA_2018_prediction.png">
</p>

At the beginning the prediction a little bit drifts away from the price but we can see further some stable trend going sideways so the time will verify whether it is a good forecast.

## Starbucks (SBUX)

Lastly lets test our model on SBUX. Its corresponding prices chart through history is as follows:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/SBUX/SBUX_whole_prices.png">
</p>

Where the last two vertical lines on the right as usual corresponds to year 2018.

First considered sample period again is 2008-2016 so lets take a look at its chart:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/SBUX/SBUX_2008_2016_prices.png">
</p>

Normalized prices for this period will be respectively:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/SBUX/SBUX_2008_2016_normalized_prices.png">
</p>

Our prediction for the whole year 2017 with 95% confidence intervals will be:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/SBUX/SBUX_2017_prediction.png">
</p>

This time our model doesn't forecast as well as previously. There are some good trend prediction in the beginning of the year but nextly it greatly underestimates the behaviour and again in the later period it gets some good forecast, mainly in the last quarter.

Nextly lets consider the 2008-2018 period including first two quarters:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/SBUX/SBUX_2008_2018_prices.png">
</p>

And our prediction for the rest of 2018 will be:

<p align="center">
  <img src="https://raw.githubusercontent.com/gdroguski/GaussianProcesses/master/Pics/SBUX/SBUX_2018_prediction.png">
</p>

At the beginning the prediction greatly overestimates the behaviour but we can see some stable trend goind upwards from about 150 day further on so it remains again for us to wait and verify what happens next.

## Summary

As said earlier the aim of this project was to learn mathematical concepts of GP and implement it later on. We saw that sometimes it forecasted prices trend surprisingly well and sometimes terrible so I wouldn't invest my own money solely relying on GP, but maybe after doing some technical analysis and finding the common results based on analysis and GP I guess why not. Potentially it could be another good technical indicator.

Mathematical definitions and theoretical descriptions were taken from positions listed in bibliography.

## Bibliography
* Bengio, Chapados, Forecasting and Trading Commodity Contract Spreads with Gaussian Processes, 2007,
* Correa, Farrell, Gaussian Process Regression Models for Predicting Stock Trends, 2007,
* Chen, Gaussian process regression methods and extensions for stock market prediction, 2017
* Rasmussen, Williams, Gaussian Processes for Machine Learning, 2006
