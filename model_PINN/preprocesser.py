import numpy as np

class Preprocesser:

    def rolling(self, df):
        roll_param = 10
        df['I'] = df['I'].rolling(roll_param).mean()
        df['Q'] = df['Q'].rolling(roll_param).mean()
        df['n'] = df['n'].rolling(roll_param).mean()
        df['x_1'] = df['x_1'].rolling(roll_param).mean()
        df['x_2'] = df['x_2'].rolling(roll_param).mean()
        df['x_3'] = df['x_3'].rolling(roll_param).mean()
        return df

    def preproc(self, df):
        df.iloc[:, :] = np.where(df < 0, 0, df)

        for col in df.columns:
            df.loc[:, col] = np.where(
                df[col] > np.quantile(df[col], 0.99),
                np.quantile(df[col], 0.99),
                df[col]
            )
        
        df = df.resample('1min', label='right', closed='right').mean()
        df = self.rolling(df)
        return df