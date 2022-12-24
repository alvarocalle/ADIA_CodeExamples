import json
from datetime import datetime
from iexfinance.refdata import get_symbols
from iexfinance.stocks import get_historical_data
import matplotlib.pyplot as plt
import pytz


def get_token():

    with open('config.json') as config_file:
        data = json.load(config_file)
        return data['TOKEN']


if __name__ == '__main__':

    tz = pytz.timezone('Europe/Madrid')
    date = datetime.now(tz).date().strftime('%Y%m%d')
    date_today = datetime.today().date()

    # IEX token
    token = get_token()
    pd_symbols = get_symbols(output_format='pandas', token=token)

    # get list of enabled tickers
    tickers = pd_symbols[pd_symbols['isEnabled']].loc[:, 'symbol']
    print(tickers)

    start = datetime(2017, 1, 1)
    end = datetime.today()

    df = get_historical_data("TSLA", start, end, output_format='pandas', token=token)
    print(df.head(5))

    df.to_csv(path_or_buf='./data/iex/TSLA.csv',
              sep='|',
              header=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    df['Close'].plot()
    plt.show()
