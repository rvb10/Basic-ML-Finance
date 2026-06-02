import os
import pdb 
import pandas as pd

class HELPER:
    def __init__(self):
        DATA_DIR = './Data'
        if not os.path.isdir(DATA_DIR):
            DATA_DIR = "../resource/asnlib/publicdata/data"

        self.DATA_DIR = DATA_DIR

    def attrRename(self, df, ticker):
        """
        Rename attributes of DataFrame
        - prepend the string "T_" to the original attribute name, where T is the string of the ticker
        """
        rename_map = {orig: ticker + "_" + orig.replace(" ", "_") for orig in df.columns.to_list()}
        return df.rename(columns=rename_map)

    def getData(self, tickers, indx, attrs1, attrs2):
        """
        Return DataFrame with data for a list of tickers plus an index
        
        Parameters
        ----------
        tickers: List
        - List of tickers
        
        indx: String
        - Ticker of the index
        
        attrs1: List
        - First set of attributes to retain
        
        attrs2: List
        - Second set of attributes to retain
        
        Returns
        -------
        DataFrame
        - attributes:
        -- each original attribute, prepended with "T_" where T is the ticker or index string
        -- This is necessary to distinguish between attributes of different tickers
        """
        DATA_DIR = self.DATA_DIR
        dateAttr = "Dt"

        # Merge both attribute lists to ensure both sets are loaded
        use_cols = list(set(attrs1 + attrs2))
        use_cols.insert(0, dateAttr)  # Ensure the date column is included

        # Read the CSV files
        dfs = []
        for ticker in tickers:
            ticker_file = os.path.join(DATA_DIR, f"{ticker}.csv")
            ticker_df = pd.read_csv(ticker_file, index_col=dateAttr, usecols=use_cols)

            # Rename attributes with ticker name
            ticker_df = self.attrRename(ticker_df, ticker)
            dfs.append(ticker_df)

        # Load the index data
        index_file = os.path.join(DATA_DIR, f"{indx}.csv")
        index_df = pd.read_csv(index_file, index_col=dateAttr, usecols=use_cols)
        index_df = self.attrRename(index_df, indx)

        dfs.append(index_df)

        # Combine all data into a single DataFrame
        data_df = pd.concat(dfs, axis=1)

        return data_df

    def renamePriceToRet(self, df, priceAttr="Adj Close"):
        rename_map = {orig: orig.replace(priceAttr.replace(" ", "_"), "Ret") for orig in df.columns.to_list()}
        return df.rename(columns=rename_map)
        
    def renamePriceToVol_Ret(self, df, volAttr="Volume"):
        rename_map = {orig: orig.replace(volAttr.replace(" ", "_"), "Vol_Change") for orig in df.columns.to_list()}
        return df.rename(columns=rename_map)

