
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import sys

def get_data(symbol,start_date='2013-01-01',end_date='2019-09-01'):
    
    df_temp = pdr.get_data_yahoo("{}".format(symbol), start=start_date,end=end_date).reset_index()
    df_temp = df_temp.drop(['Close'],1)
    df_temp = df_temp.rename(columns={'Adj Close':'Close'})
    df_temp_SPY = pdr.get_data_yahoo("SPY", start=start_date).reset_index()
    df_SPY = df_temp_SPY[['Date']]
    df = df_SPY.merge(df_temp,left_on='Date',right_on="Date")
    df = df.set_index('Date')
    del df_temp, df_temp_SPY, df_SPY

    return df

def standardize_data(df):
    std_scaler = StandardScaler()
    df_scaled = std_scaler.fit_transform(df)
    out = pd.DataFrame(df_scaled,index=df.index)
    out.columns = df.columns
    
    return out

def features_target_generation(df, num_days_ahead):
    
    features = pd.DataFrame(index=df.index).sort_index()
    outcomes = pd.DataFrame(index=df.index)
    
    features['f01'] = df.Close / df.Open - 1 
    features['f02'] = df.Open / df.Close.shift(1) - 1 
    features['f03'] = np.log(df.Volume) 
    features['f04'] = df.Volume.diff() 
    features['f05'] = df.Volume.pct_change()
    features['f06'] = df.Volume.rolling(5).mean().apply(np.log)
    features['f07'] = df.Volume.rolling(30,min_periods=20).mean().apply(np.log)
    features['f08'] = df.Volume.rolling(180,min_periods=20).mean().apply(np.log)

    x = df.Close
    features['f09'] = (x - x.rolling(180,min_periods=20).mean()) / x.rolling(180,min_periods=20).std()
    
    x = df.Close
    features['f10'] = (x - x.rolling(5).mean()) / x.rolling(window=5).std()
        
    x = df.Close
    features['f11'] = (x - x.rolling(30,min_periods=20).mean()) / x.rolling(30,min_periods=20).std()

    features['f12'] = df.Close
    features['f13'] = df.Close.shift(1)
    features['f14'] = df.Close.shift(7)
    features['f15'] = df.Close.shift(30)
    features['f16'] = df.Close.shift(90)
    features['f17'] = df.Close.shift(180)
    features['f18'] = df.Close.shift(270)
    
    x = df.Volume
    features['f31'] = x.rolling(180,min_periods=20).apply(lambda x: pd.Series(x).rank(pct=True)[0])
    
    n_bins=10
    x = df.Volume
    features['f32'] = x.rolling(180,min_periods=20).apply(lambda x: pd.qcut(pd.Series(x), q=n_bins, labels = range(1,n_bins+1))[0])
    
    features['f33'] = features['f05'].apply(np.sign)
    
    x = features['f33']
    features['f34'] = abs(x.groupby((x!=x.shift(1)).cumsum()).cumsum())
     
    features = standardize_data(features.dropna())
    
    begin_num = 35
    one_hot_frame=pd.DataFrame(pd.get_dummies(df.index.month))
    one_hot_frame.index = df.index
    feat_names = ['f'+str(num) for num in list(range(begin_num,begin_num+12,1))]
    one_hot_frame.columns = feat_names
    features=features.join(one_hot_frame)
    
    # generating outcomes dataframe
    outcomes = df.Close.shift(-1 * int(num_days_ahead))
    outcomes = outcomes.to_frame().dropna()
    outcomes.rename(columns = {'Close':'target'}, inplace = True)
    
    # returning X and y sets for the model
    temp=features.join(outcomes,how='inner')
    X = temp[features.columns]
    y = temp['target']
    
    del temp, features, outcomes
    return X, y

def LR_model(X,y):

    recalc_dates = X.resample('Q').mean().index.values[:-1]
    index = X.columns

    # fitting LR model
       
    models_LR = pd.Series(index=recalc_dates)
    for date in recalc_dates:
        X_train = X.loc[date-pd.Timedelta('270 days'):date,:]
        y_train = y.loc[date-pd.Timedelta('270 days'):date]
        model = LinearRegression()
        model.fit(X_train, y_train)
        models_LR.loc[date] = model

    # predicting out of sample values for the future day stock prices

    begin_dates = models_LR.index
    end_dates = models_LR.index[1:].append(pd.to_datetime(['31-12-2020']))
    predictions_LR = pd.Series(index=X.index)

    for i,model in enumerate(models_LR):
        X_test = X.loc[begin_dates[i]:end_dates[i],:]
        preds = pd.Series(model.predict(X_test),index=X_test.index)
        predictions_LR.loc[X_test.index]=preds
        
    LR_idx = y.dropna().index.intersection(predictions_LR.dropna().index)
    rsq_LR = r2_score(y[LR_idx],predictions_LR[LR_idx])

    return rsq_LR, LR_idx, predictions_LR



def SVR_model(X,y):

    def grid_search_SVR(X,y):
        from sklearn.model_selection import TimeSeriesSplit
        param_grid={
                'C': [0.1, 1, 100, 1000, 10000, 100000, 1000000, 10000000],
                'epsilon': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5],
                'gamma': [0.00001, 0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
            },
        cv = TimeSeriesSplit(n_splits = 5)
        grid = GridSearchCV(SVR(), param_grid = param_grid, cv = cv)
        grid.fit(X,y)
                
        return grid.best_estimator_

    # Walk forward SVR model training based on the rolling window of 270 days

    recalc_dates = X.resample('Q').mean().index.values[:-1]
    index = X.columns

    # fitting SVR models
       
    models_SVR = pd.Series(index=recalc_dates)
    for date in recalc_dates:
        X_train = X.loc[date - pd.Timedelta('270 days'):date,:]
        y_train = y.loc[date - pd.Timedelta('270 days'):date]
        model = grid_search_SVR(X_train,y_train)
        models_SVR.loc[date] = model

    # predicting out of sample values for the next day stock prices

    begin_dates = models_SVR.index
    end_dates = models_SVR.index[1:].append(pd.to_datetime(['31-12-2020']))
    predictions_SVR = pd.Series(index = X.index)

    for i,model in enumerate(models_SVR):
        X_test = X.loc[begin_dates[i]:end_dates[i],:]
        preds = pd.Series(model.predict(X_test),index = X_test.index)
        predictions_SVR.loc[X_test.index] = preds
        
    SVR_idx = y.dropna().index.intersection(predictions_SVR.dropna().index)
    rsq_SVR = r2_score(y[SVR_idx],predictions_SVR[SVR_idx])

    return rsq_SVR, SVR_idx, predictions_SVR



def random_forest_model(X,y):

    def grid_search_RFR(X,y):
        from sklearn.model_selection import TimeSeriesSplit
        param_grid={
                'n_estimators': [100],
                'max_features'      : ["auto", "sqrt", "log2"],
                'min_samples_split' : [2,4,8],
                'bootstrap': [True, False],
                'random_state': [0]
            },
        cv = TimeSeriesSplit(n_splits = 5)
        grid = GridSearchCV(RandomForestRegressor(), param_grid = param_grid, cv = cv)
        grid.fit(X,y)
                
        return grid.best_estimator_

    # Walk forward random forest model training based on the rolling window of 270 days

    recalc_dates = X.resample('Q').mean().index.values[:-1]
    index = X.columns

    # fitting random forest models
       
    models_RFR = pd.Series(index=recalc_dates)
    for date in recalc_dates:
        X_train = X.loc[date - pd.Timedelta('270 days'):date,:]
        y_train = y.loc[date - pd.Timedelta('270 days'):date]
        model = grid_search_RFR(X_train,y_train)
        models_RFR.loc[date] = model

    # predicting out of sample values for the next day stock prices

    begin_dates = models_RFR.index
    end_dates = models_RFR.index[1:].append(pd.to_datetime(['31-12-2020']))
    predictions_RFR = pd.Series(index = X.index)

    for i,model in enumerate(models_RFR):
        X_test = X.loc[begin_dates[i]:end_dates[i],:]
        preds = pd.Series(model.predict(X_test),index = X_test.index)
        predictions_RFR.loc[X_test.index] = preds
        
    RFR_idx = y.dropna().index.intersection(predictions_RFR.dropna().index)
    rsq_RFR = r2_score(y[RFR_idx],predictions_RFR[RFR_idx])

    return rsq_RFR, RFR_idx, predictions_RFR



def main():

    symbol = sys.argv[1]
    start_date =  sys.argv[2]
    end_date = sys.argv[3]
    num_days_ahead = sys.argv[4]
    df = get_data(symbol, start_date, end_date)
    X, y = features_target_generation(df, num_days_ahead)
    rsq_LR, LR_idx, predictions_LR = LR_model(X,y)
    rsq_SVR, SVR_idx, predictions_SVR = SVR_model(X,y)
    rsq_RFR, RFR_idx, predictions_RFR = random_forest_model(X,y)
    ig, ax = plt.subplots(figsize=(14,8))
    idx_sample = LR_idx[-100:]
    ax.plot(idx_sample, y[idx_sample], 'g', label='True')
    ax.plot(idx_sample, predictions_LR[idx_sample], '--r', label='LR, RSQ = {}'.format(rsq_LR))
    ax.plot(idx_sample, predictions_SVR[idx_sample], '--b', label='SVR, RSQ = {}'.format(rsq_SVR))
    ax.plot(idx_sample, predictions_RFR[idx_sample], '--y', label='Random Forest, RSQ = {}'.format(rsq_RFR))
    plt.title('Predicted vs true stock prices')
    leg = ax.legend();
    plt.show()
    
if __name__ == "__main__":
    main()