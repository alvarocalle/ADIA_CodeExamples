from features.indicators import features
import pandas as pd


def main():

    df = pd.read_csv('../data/kaggle/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv', sep=',', header=0,
                     names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Volume_Currency', 'Weighted_Price'])

    # fill NAs
    df = df.fillna(method='bfill').reset_index(drop=True)

    # add features
    features_df = features(df)

    # save features
    features_df.to_csv('../data/processed/features.csv', index=False)


if __name__ == "__main__":
    # execute only if run as a script
    main()
