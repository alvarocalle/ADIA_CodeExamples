import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib import style
from datetime import datetime

# finance module is no longer part of matplotlib
# see: https://github.com/matplotlib/mpl_finance
from mpl_finance import candlestick_ochl as candlestick

# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
style.use('dark_background')
#style.use('seaborn')

# Dates for BTC Trading are in unix timestamp format
# we do not need to do date2num transformation as in StockTradingGraph.py

VOLUME_CHART_HEIGHT = 0.33

UP_COLOR = '#02AC66'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#02AC66'
DOWN_TEXT_COLOR = '#EF534F'
OFFSET = 0


class BitcoinTradingGraph:
    """
    BTC trading visualization using matplotlib made to render OpenAI gym environments

    The green ghosted tags represent buys of BTC and the red ghosted tags represent sells.
    The white tag on the top right is the agent's current net worth and the bottom right tag is the current price of Bitcoin
    """

    def __init__(self, df, title=None):
        self.df = df
        self.net_worths = np.zeros(len(df['Timestamp']))

        # Create a figure on screen and set the title
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle(title)

        # Create top subplot for net worth axis
        self.net_worth_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)

        # Create bottom subplot for shared price/volume axis
        self.price_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_worth_ax)

        # Create a new axis for volume which shares its x-axis with price
        self.volume_ax = self.price_ax.twinx()

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_net_worth(self, current_step, net_worth, step_range, dates):
        """
        Print net worth as timeseries together with annotated wealth
        :param current_step:
        :param net_worth:
        :param step_range:
        :param dates:
        :return:
        """

        # Clear the frame rendered last step
        self.net_worth_ax.clear()

        # Plot net worth
        self.net_worth_ax.plot_date(
            dates, self.net_worths[step_range], '-', lw=1, color='white', label='Net Worth')

        # Show legend, which uses the label we defined for the plot above
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = self.df['Timestamp'].values[current_step]
        last_net_worth = self.net_worths[current_step]

        # Annotate the current net worth on the net worth graph
        self.net_worth_ax.annotate('{0:.2f}'.format(net_worth), (last_date+OFFSET, last_net_worth),
                                   xytext=(last_date, last_net_worth),
                                   bbox=dict(boxstyle='round', fc='k', ec='w', lw=1),
                                   color="white",
                                   fontsize="small")

        # Add space above and below min/max net worth
        self.net_worth_ax.set_ylim(
            min(self.net_worths[np.nonzero(self.net_worths)]) / 1.25, max(self.net_worths) * 1.25)

    def _render_price(self, current_step, dates, step_range):
        """
        Print prices as candlesticks together with annotated current price
        :param current_step:
        :param dates:
        :param step_range:
        :return:
        """
        self.price_ax.clear()

        # Format data for OHCL candlestick graph
        candlesticks = zip(dates,
                           self.df['Open'].values[step_range], self.df['Close'].values[step_range],
                           self.df['High'].values[step_range], self.df['Low'].values[step_range])

        # Plot price using candlestick graph from mpl_finance
        candlestick(self.price_ax, candlesticks, width=5,
                    colorup=UP_COLOR, colordown=DOWN_COLOR)

        last_date = self.df['Timestamp'].values[current_step]
        last_close = self.df['Close'].values[current_step]
        last_high = self.df['High'].values[current_step]

        # Print the current price to the price axis
        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date+OFFSET, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                                         fc="black", ec="white", pad=0.2, lw=1),
                               color="white",
                               fontsize="small")

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])
                               * VOLUME_CHART_HEIGHT, ylim[1])

    def _render_volume(self, current_step, net_worth, dates, step_range):
        self.volume_ax.clear()

        volume = np.array(self.df['Volume_BTC'].values[step_range])

        pos = self.df['Open'].values[step_range] - \
              self.df['Close'].values[step_range] < 0
        neg = self.df['Open'].values[step_range] - \
              self.df['Close'].values[step_range] > 0

        # Color volume bars based on price direction on that date
        self.volume_ax.bar(dates[pos], volume[pos], color=UP_COLOR,
                           alpha=0.5, width=10, align='center')
        self.volume_ax.bar(dates[neg], volume[neg], color=DOWN_COLOR,
                           alpha=0.5, width=10, align='center')

        # Cap volume axis height below price chart and hide ticks
        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])

    def _render_trades(self, trades, step_range):
        for trade in trades:
            if trade['step'] in step_range:
                date = self.df['Timestamp'].values[trade['step']]
                high = self.df['High'].values[trade['step']]
                low = self.df['Low'].values[trade['step']]

                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_TEXT_COLOR
                else:
                    high_low = high
                    color = DOWN_TEXT_COLOR

                total = '{0:.2f}'.format(trade['total'])

                # Print the current price to the price axis
                self.price_ax.annotate(f'${total}', (date, high_low),
                                       xytext=(date, high_low),
                                       color=color,
                                       fontsize=8,
                                       arrowprops=(dict(color=color, headwidth=10, headlength=8)))

    def render(self, current_step, net_worth, trades, window_size=40):
        self.net_worths[current_step] = net_worth

        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step + 1)

        # Format dates as timestamps, necessary for candlestick graph
        dates = np.array([x for x in self.df['Timestamp'].values[step_range]])

        self._render_net_worth(current_step, net_worth, step_range, dates)
        self._render_price(current_step, dates, step_range)
        self._render_volume(current_step, net_worth, dates, step_range)
        self._render_trades(trades, step_range)

        # Format the date ticks to be more easily read
        formatter = DateFormatter("%Y-%m-%d %H:%M")
        human_dates = [datetime.fromtimestamp(ts) for ts in dates]
        self.price_ax.set_xticklabels(human_dates,  #self.df['Timestamp'].values[step_range],
                                      rotation=45,
                                      horizontalalignment='right')
        #self.price_ax.xaxis.set_major_formatter(formatter)

        # Hide duplicate net worth date labels
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)

        # pause between rendered frames
        plt.pause(0.01)

    def close(self):
        plt.close()
