import os
import pandas as pd

class HELPER:
    def __init__(self):
        self.DATA_DIR = './Data'
        if not os.path.isdir(self.DATA_DIR):
            self.DATA_DIR = "../resource/asnlib/publicdata/data"

    def attrRename(self, df, ticker):
        rename_map = {col: f"{ticker}_{col.replace(' ', '_')}" for col in df.columns}
        return df.rename(columns=rename_map)

    def getData(self, tickers, attrs, subdir="train"):
        use_cols = ["Dt"] + attrs
        dfs = []
        
        full_path = os.path.join(self.DATA_DIR, subdir)
        
        for ticker in tickers:
            file_path = os.path.join(full_path, f"{ticker}.csv")
            df = pd.read_csv(file_path, usecols=use_cols, index_col="Dt")
            df = self.attrRename(df, ticker)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, axis=1)
        return combined_df

    def renamePriceToRet(self, df, priceAttr="Adj Close"):
        old = priceAttr.replace(" ", "_")
        new_names = {col: col.replace(old, "Ret") for col in df.columns if old in col}
        return df.rename(columns=new_names)

    def load_stock_data(self, ticker="AAPL", subdir="train", lag_days=5):

        file_path = os.path.join(self.DATA_DIR, subdir, f"{ticker}.csv")
        df = pd.read_csv(file_path)
        
        if 'Dt' not in df.columns:
            raise ValueError("Expected 'Dt' (date) column in CSV.")
        
        df['Return'] = df['Close'].pct_change()

        # Create lag features
        for lag in range(1, lag_days + 1):
            df[f'Lag_{lag}'] = df['Return'].shift(lag)

        df.dropna(inplace=True)

        X = df[[f'Lag_{i}' for i in range(1, lag_days + 1)]]
        y = df['Return']
        dates = df['Dt']

        return X, y, dates
