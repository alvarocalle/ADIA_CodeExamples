# ---------------------------------------------------------------------------
# Technical Indicators
# takes BTC prices and creates a new csv file where indicators are included
# this is done to speed up the trading script. we do not overload with feature creation
# Assumes input DataFrame without NAs
# ta-lib (python wrapper) https://github.com/mrjbq7/ta-lib
# ta (other TA library in Python): https://pypi.org/project/ta/
#
# The selection of features has been carried out in a study in
# notebooks/RLTrading - Technical Analysis.ipynb
#
# -------------------------------------------------------------------------------------


import talib
import ta
import pandas as pd


def features(df) -> pd.DataFrame:
    """
    Function that creates the set of features.
    Makes use of TA-lib and ta to define the different indicators.

    :param df: dataset with bars data in OCHL format
    :output dataset: (dataframe) with constructed features and target variable (classes)
    """

    # Momentum Indicator Functions
    momentum_df = pd.DataFrame()

    momentum_df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
    # momentum_df['ADXR'] = talib.ADXR(df['High'], df['Low'], df['Close'])
    momentum_df['APO'] = talib.APO(df['Close'])
    momentum_df['AROONDOWN'], df['AROONUP'] = talib.AROON(df['High'], df['Low'])
    # momentum_df['AROONOSC'] = talib.AROONOSC(df['High'], df['Low'])
    momentum_df['BOP'] = talib.BOP(df['Open'], df['High'], df['Low'], df['Close'])
    # momentum_df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
    # momentum_df['CMO'] = talib.CMO(df['Close'])
    # momentum_df['DX'] = talib.DX(df['High'], df['Low'], df['Close'])
    # momentum_df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(df['Close'])
    # momentum_df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
    # momentum_df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'])
    momentum_df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'])
    # momentum_df['MOM'] = talib.MOM(df['Close'])
    # momentum_df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'])
    # momentum_df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'])
    # momentum_df['PPO'] = talib.PPO(df['Close'])
    momentum_df['ROC'] = talib.ROC(df['Close'])
    # momentum_df['ROCP'] = talib.ROCP(df['Close'])
    # momentum_df['ROCR'] = talib.ROCR(df['Close'])
    # momentum_df['RSI'] = talib.RSI(df['Close'])
    # momentum_df['SLOWK'], df['SLOWD'] = talib.STOCH(df['High'], df['Low'], df['Close'])
    # momentum_df['FASTK'], df['FASTD'] = talib.STOCHF(df['High'], df['Low'], df['Close'])
    momentum_df['TRIX'] = talib.TRIX(df['Close'])
    # momentum_df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'])
    # momentum_df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'])
    # momentum_df['TSI'] = ta.momentum.tsi(df['Close'])
    # momentum_df['UO'] = ta.momentum.uo(df['High'], df['Low'], df['Close'])
    # momentum_df['AO'] = ta.momentum.ao(df['High'], df['Close'])

    # Overlap Studies Functions
    overlap_df = pd.DataFrame()

    overlap_df['BBH'], df['BBM'], df['BBL'] = talib.BBANDS(df['Close'])
    # overlap_df['DEMA'] = talib.DEMA(df['Close'])  #equivalent to TSI
    # overlap_df['EMA'] = talib.EMA(df['Close'])
    # overlap_df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['Close'])
    # overlap_df['KAMA'] = talib.KAMA(df['Close'])
    # overlap_df['MA'] = talib.MA(df['Close'])
    # overlap_df['MAMA'], df['FAMA'] = talib.MAMA(df['Close'])
    # overlap_df['MIDPOINT'] = talib.MIDPOINT(df['Close'])
    # overlap_df['MIDPRICE'] = talib.MIDPRICE(df['High'], df['Low'])
    # overlap_df['SAR'] = talib.SAR(df['High'], df['Low'])
    overlap_df['SAREXT'] = talib.SAREXT(df['High'], df['Low'])
    # overlap_df['SMA'] = talib.SMA(df['Close'])
    # overlap_df['T3'] = talib.T3(df['Close'])
    # overlap_df['TEMA'] = talib.TEMA(df['Close'])
    # overlap_df['TRIMA'] = talib.TRIMA(df['Close'])
    # overlap_df['WMA'] = talib.WMA(df['Close'])

    # Trend
    trend_df = pd.DataFrame()

    trend_df['VIP'] = ta.trend.vortex_indicator_pos(df['High'], df['Low'], df['Close'])
    trend_df['VIN'] = ta.trend.vortex_indicator_neg(df['High'], df['Low'], df['Close'])
    # trend_df['VID'] = trend_df['VIP'] - trend_df['VIN']
    trend_df['MI'] = ta.trend.mass_index(df['High'], df['Low'])
    trend_df['DPO'] = ta.trend.dpo(df['Close'])
    trend_df['KST'] = ta.trend.kst(df['Close'])
    # trend_df['ICHIMOKU_A'] = ta.trend.ichimoku_a(df['High'], df['Low'])
    # trend_df['ICHIMOKU_B'] = ta.trend.ichimoku_b(df['High'], df['Low'])

    # Pattern Recognition Functions [Japanese candlesticks]
    pattern_df = pd.DataFrame()

    pattern_df['CDL2CROWS'] = talib.CDL2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDL3INSIDE'] = talib.CDL3INSIDE(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLBELTHOLD'] = talib.CDLBELTHOLD(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLDOJI'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLDOJISTAR'] = talib.CDLDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLENGULFING'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLHAMMER'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLHARAMI'] = talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLHIKKAKE'] = talib.CDLHIKKAKE(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLINNECK'] = talib.CDLINNECK(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLKICKING'] = talib.CDLKICKING(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLLONGLINE'] = talib.CDLLONGLINE(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLMARUBOZU'] = talib.CDLMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDLMATHOLD'] = talib.CDLMATHOLD(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLONNECK'] = talib.CDLONNECK(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLPIERCING'] = talib.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLSHORTLINE'] = talib.CDLSHORTLINE(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLTAKURI'] = talib.CDLTAKURI(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLTHRUSTING'] = talib.CDLTHRUSTING(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLTRISTAR'] = talib.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    # 3pattern_df['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(df['Open'], df['High'], df['Low'], df['Close'])
    pattern_df['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    # pattern_df['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(df['Open'], df['High'], df['Low'], df['Close'])

    # Volatility Indicator Functions
    volatility_df = pd.DataFrame()

    volatility_df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
    volatility_df['NATR'] = talib.NATR(df['High'], df['Low'], df['Close'])
    # volatility_df['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])
    volatility_df['KCHI'] = ta.volatility.keltner_channel_hband_indicator(df['High'], df['Low'], df['Close'])
    # volatility_df['KCLI'] = ta.volatility.keltner_channel_lband_indicator(df['High'], df['Low'], df['Close'])
    # volatility_df['DCHI'] = ta.volatility.donchian_channel_hband_indicator(df['Close'])
    # volatility_df['DCLI'] = ta.volatility.donchian_channel_lband_indicator(df['Close'])

    # Volume Indicator Functions
    volume_df = pd.DataFrame()

    volume_df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    volume_df['ADOSC'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
    # volume_df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    # volume_df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
    volume_df['FI'] = ta.volume.force_index(df['Close'], df['Volume'])
    volume_df['EM'] = ta.volume.ease_of_movement(df['High'], df['Low'], df['Close'], df['Volume'])
    volume_df['VPT'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
    # volume_df['NVI'] = ta.volume.negative_volume_index(df['Close'], df['Volume'])

    # Remove NAs from DataFrames:
    momentum_df = momentum_df.fillna(method='bfill').reset_index(drop=True)
    overlap_df = overlap_df.fillna(method='bfill').reset_index(drop=True)
    trend_df = trend_df.fillna(method='bfill').reset_index(drop=True)
    pattern_df = pattern_df.fillna(method='bfill').reset_index(drop=True)
    volatility_df = volatility_df.fillna(method='bfill').reset_index(drop=True)
    volume_df = volume_df.fillna(method='bfill').reset_index(drop=True)

    # Merge all DataFrames
    df = pd.concat([momentum_df,
                    overlap_df,
                    trend_df,
                    pattern_df,
                    volatility_df,
                    volume_df], axis=1)
    return df
