import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import data_handler


class Wrapper:
    __company_data = None
    __prices_data = None
    __quarters = None
    __max_days = None
    __alpha = None
    __iterations = None
    __kernels = None
    __gp = None

    def __init__(self, company_name: str):
        self.__company_data = data_handler.CsvHandler(company_name)
        self.__prices_data = self.__company_data.get_equal_length_prices()
        self.__quarters = self.__company_data.quarters
        self.__years = self.__company_data.years
        self.__max_days = self.__company_data.max_days

        kernel = 63 * RBF(length_scale=1)
        self.__alpha = 1e-10
        self.__iterations = 10
        self.__kernels = [kernel]
        self.__gp = GaussianProcessRegressor(kernel=self.__kernels[0], alpha=self.__alpha,
                                             n_restarts_optimizer=self.__iterations,
                                             normalize_y=False)

    def get_eval_model(self, start_year: int, end_year: int, pred_year: int, pred_quarters: list = None):
        years_quarters = list(range(start_year, end_year + 1)) + ['Quarter']
        training_years = years_quarters[:-2]
        df_prices = self.__prices_data[self.__prices_data.columns.intersection(years_quarters)]

        possible_days = list(df_prices.index.values)
        X = np.empty([1,2], dtype=int)
        Y = np.empty([1], dtype=float)

        first_year_prices = df_prices[start_year]
        if start_year == self.__company_data.years[0]:
            first_year_prices = (first_year_prices[first_year_prices.iloc[:] != 0])
            first_year_prices = (pd.Series([0.0], index=[first_year_prices.index[0]-1])).append(first_year_prices)

        first_year_days = list(first_year_prices.index.values)
        first_year_X = np.array([[start_year, day] for day in first_year_days])

        X = first_year_X
        Y = np.array(first_year_prices)
        for current_year in training_years[1:]:
            current_year_prices = list(df_prices.loc[:, current_year])
            current_year_X = np.array([[current_year, day] for day in possible_days])
            X = np.append(X, current_year_X, axis=0)
            Y = np.append(Y, current_year_prices)

        last_year_prices = df_prices[end_year]
        last_year_prices = last_year_prices[last_year_prices.iloc[:].notnull()]

        last_year_days = list(last_year_prices.index.values)
        if pred_quarters is not None:
            length = 63 * (pred_quarters[0] - 1)
            last_year_days = last_year_days[:length]
            last_year_prices = last_year_prices[:length]
        last_year_X = np.array([[end_year, day] for day in last_year_days])

        X = np.append(X, last_year_X, axis=0)
        Y = np.append(Y, last_year_prices)

        if pred_quarters is not None:
            pred_days = [day for day in
                         range(63 * (pred_quarters[0]-1), 63 * pred_quarters[int(len(pred_quarters) != 1)])]
        else:
            pred_days = list(range(0, self.__max_days))
        x_mesh = np.linspace(pred_days[0], pred_days[-1]
                             , 2000)
        x_pred = ([[pred_year, x_mesh[i]] for i in range(len(x_mesh))])

        self.__gp = self.__gp.fit(X, Y)
        self.__kernels.append(self.__gp.kernel_)

        y_mean, y_cov = self.__gp.predict(x_pred, return_cov=True)

        return x_mesh, y_mean, y_cov

    def get_kernels(self):
        return self.__kernels
