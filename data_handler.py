import pandas as pd
import numpy as np


class CsvHandler:
    df = None
    quarters = None
    years = None
    max_days = None

    def __init__(self, csv_name: str):
        self.__load_data(csv_name)
        self.df['Norm Adj Close'] = self.__add_normalized_data(self.df)
        self.df['Quarter'] = self.__add_quarters(self.df)
        self.max_days = 252

    def get_equal_length_prices(self, normalized=True):
        df = self.__shift_first_year_prices()
        for i in range(1, len(self.years)):
            df = pd.concat([df, pd.DataFrame(self.get_year_data(year=self.years[i], normalized=normalized))], axis=1)

        df = df[:self.max_days]

        quarters = []
        for j in range(0, len(self.quarters)):
            for i in range(0, self.max_days // 4):
                quarters.append(self.quarters[j])
        quarters = pd.DataFrame(quarters)

        df = pd.concat([df, quarters], axis=1)
        df.columns = self.years + ['Quarter']
        df.index.name = 'Day'

        self.__fill_last_rows(df)

        return df

    def get_year_data(self, year: int, normalized=True):
        if year not in self.years:
            raise ValueError('\n' +
                             'Input year: {} not in available years: {}'.format(year, self.years))

        prices = (self.df.loc[self.df['Date'].dt.year == year])
        if normalized:
            return np.asarray(prices.loc[:, 'Norm Adj Close'])
        else:
            return np.asarray(prices.loc[:, 'Adj Close'])

    def get_whole_prices(self, start_year: int, end_year: int):
        if start_year < self.years[0] or end_year > self.years[-1]:
            raise ValueError('\n' +
                             'Input years out of available range! \n' +
                             'Max range available: {}-{}\n'.format(self.years[0], self.years[-1]) +
                             'Was: {}-{}'.format(start_year, end_year))

        df = (self.df.loc[(self.df['Date'].dt.year >= start_year) & (self.df['Date'].dt.year <= end_year)])
        df = df.loc[:, ['Date', 'Adj Close']]

        return df

    def show(self, max_rows=None, max_columns=None):
        with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns):
            print(self.df)

    def __load_data(self, csv_name: str):
        self.df = pd.read_csv('Data/' + csv_name + '.csv')
        self.df = self.df.iloc[:, [0, 5]]
        self.df = self.df.dropna()
        self.df.Date = pd.to_datetime(self.df.Date)
        self.quarters = ['Q' + str(i) for i in range(1, 5)]

    def __add_normalized_data(self, df):
        normalized = pd.DataFrame()

        self.years = list(df.Date)
        self.years = list({self.years[i].year for i in range(0, len(self.years))})

        for i in range(0, len(self.years)):
            prices = self.get_year_data(year=self.years[i], normalized=False)
            mean = np.mean(prices)
            std = np.std(prices)
            prices = [(prices[i] - mean) / std for i in range(0, len(prices))]
            prices = [(prices[i] - prices[0]) for i in range(0, len(prices))]
            normalized = normalized.append(prices, ignore_index=True)

        return normalized

    def __add_quarters(self, df):
        quarters = pd.DataFrame()

        for i in range(0, len(self.years)):
            dates = list((df.loc[df['Date'].dt.year == self.years[i]]).iloc[:, 0])
            dates = pd.DataFrame([self.__get_quarter(dates[i].month) for i in range(0, len(dates))])
            quarters = quarters.append(dates, ignore_index=True)

        return quarters

    def __get_quarter(self, month: int):
        return self.quarters[(month - 1) // 3]

    def __shift_first_year_prices(self):
        prices = pd.DataFrame(self.get_year_data(self.years[0]))
        df = pd.DataFrame([0 for _ in range(self.max_days - len(prices.index))])
        df = pd.concat([df, prices], ignore_index=True)

        return df

    def __fill_last_rows(self, df):
        years = self.years[:-1]

        for year in years:
            mean = np.mean(df[year])
            for i in range(self.max_days - 1, -1, -1):
                current_price = df.iloc[i, df.columns.get_loc(year)]
                if np.isnan(current_price):
                    df.iloc[i, df.columns.get_loc(year)] = mean
                else:
                    break
