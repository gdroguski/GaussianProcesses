# Gaussian Process regression and forecasting stock trends

The aim of this project was to learn the mathematical concepts of Gaussian Processes and implement them later on in real-world problems - in adjusted closing prices prediction consisted of three selected stock prices. 

It is obvious that the method developped during this process of creation is not ideal, if it were so I wouldn't share this publictly and made profits myself instead. ;) But nevertheless it can give some good forecasts and be used as another indicator in technical analysis of stock prices as we will see later on below.

## Gaussian Processes

Gaussian processes (Rasmussen and Williams 2006) are a general and flexible class of models for nonlinear regression and classification. They have received attention in the machine learning community over last years, having originally been introduced in geostatistics. They differ from neural networks in that they engage in a full Bayesian treatment, supplying a complete posterior distribution of forecasts. For regression, they are also computationally relatively simple to implement, the basic model requiring only solving a system of linear equations with computational complexity <img src="https://rawgit.com/gdroguski/GaussianProcesses/readme_stuff/Pics/Latex//90846c243bb784093adbb6d2d0b2b9d0.svg?invert_in_darkmode" align=middle width=43.022265pt height=26.76201000000001pt/>.

...