# fundamental modules
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import joblib
import json
import sys
import os
from pathlib import Path
# preprocessing
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
from sklearn.base import clone
import xgboost
import shap
from tune_sklearn import TuneSearchCV, TuneGridSearchCV
from scipy.stats import randint
# metrics 
import mlfinlab as ml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    log_loss,
    )
from boruta import BorutaPy
# finance packages
import trademl as tml
# import vectorbt as vbt

### DON'T SHOW GRAPH OPTION
matplotlib.use("Agg")


### GLOBALS
DATA_PATH = 'D:/market_data/usa/ohlcv_features'

### NON-MODEL HYPERPARAMETERS
num_threads = 1
structural_break_regime = 'all'
labeling_technique = 'trend_scanning'
std_outlier = 10
tb_volatility_lookback = 500
tb_volatility_scaler = 1
tb_triplebar_num_days = 10
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.004
ts_look_forward_window = 1200  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
sample_weights_type = 'returns'
cv_type = 'purged_kfold'
cv_number = 4
rand_state = 3
stationary_close_lables = False

### MODEL HYPERPARAMETERS
# max_depth = 3
# max_features = 20
# n_estimators = 500

### POSTMODEL PARAMETERS
keep_important_features = 25
# vectorbt_slippage = 0.0015
# vectorbt_fees = 0.0015


def import_data(data_path, remove_cols, contract='SPY'):
    # import data
    with pd.HDFStore(data_path + '/' + contract + '.h5') as store:
        data = store.get(contract)
    data.sort_index(inplace=True)
    
    # remove variables
    remove_cols = [col for col in remove_cols if col in data.columns]
    data.drop(columns=remove_cols, inplace=True)
    
    return data


if __name__ == '__main__':
    
    ### IMPORT DATA
    remove_ohl = ['open', 'low', 'high', 'average', 'barCount',
                  'open_vix', 'high_vix', 'low_vix', 'close_vix', 'volume_vix',
                  'open_orig', 'high_orig', 'low_orig']
    data = import_data(DATA_PATH, remove_ohl, contract='SPY')


    ### REGIME DEPENDENT ANALYSIS
    if structural_break_regime == 'chow':
        if (data.loc[data['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
            data = data.iloc[-(60*8*365):]
        else:
            data = data.loc[data['chow_segment'] == 1]

    ### USE STATIONARY CLOSE TO CALCULATE LABELS
    if stationary_close_lables:
        data['close_orig'] = data['close']  # with original close reslts are pretty bad!


    ### REMOVE OUTLIERS
    # outlier_remove = tml.modeling.pipelines.OutlierStdRemove(std_outlier)
    # data = outlier_remove.fit_transform(data)


    ### LABELING
    if labeling_technique == 'triple_barrier':
        # TRIPLE BARRIER LABELING
        triple_barrier_pipe= tml.modeling.pipelines.TripleBarierLabeling(
            close_name='close_orig',
            volatility_lookback=tb_volatility_lookback,
            volatility_scaler=tb_volatility_scaler,
            triplebar_num_days=tb_triplebar_num_days,
            triplebar_pt_sl=tb_triplebar_pt_sl,
            triplebar_min_ret=tb_triplebar_min_ret,
            num_threads=num_threads,
            tb_min_pct=tb_min_pct
        )
        tb_fit = triple_barrier_pipe.fit(data)
        labeling_info = tb_fit.triple_barrier_info
        X = tb_fit.transform(data)
    elif labeling_technique == 'trend_scanning':
        trend_scanning_pipe = tml.modeling.pipelines.TrendScanning(
            close_name='close_orig',
            volatility_lookback=tb_volatility_lookback,
            volatility_scaler=tb_volatility_scaler,
            ts_look_forward_window=ts_look_forward_window,
            ts_min_sample_length=ts_min_sample_length,
            ts_step=ts_step
            )
        labeling_info = trend_scanning_pipe.fit(data)
        X = trend_scanning_pipe.transform(data)
    elif labeling_technique == 'fixed_horizon':
        X = data.copy()
        labeling_info = ml.labeling.fixed_time_horizon(data['close_orig'], threshold=0.005, resample_by='B').dropna().to_frame()
        labeling_info = labeling_info.rename(columns={'close_orig': 'bin'})
        print(labeling_info.iloc[:, 0].value_counts())
        X = X.iloc[:-1, :]


    ### CLUSTERED FEATURES
    # feat_subs = ml.clustering.feature_clusters.get_feature_clusters(
    #     X, dependence_metric='information_variation',
    #     distance_metric='angular', linkage_method='singular',
    #     n_clusters=1)


    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns=['close_orig']), labeling_info['bin'],
        test_size=0.10, shuffle=False, stratify=None)


    ### SAMPLE WEIGHTS (DECAY FACTOR CAN BE ADDED!)
    if sample_weights_type == 'returns':
        sample_weigths = ml.sample_weights.get_weights_by_return(
            labeling_info.reindex(X_train.index),
            data.loc[X_train.index, 'close_orig'],
            num_threads=1)
    elif sample_weights_type == 'time_decay':
        sample_weigths = ml.sample_weights.get_weights_by_time_decay(
            labeling_info.reindex(X_train.index),
            data.loc[X_train.index, 'close_orig'],
            decay=0.5, num_threads=1)
    elif labeling_technique is 'trend_scanning':
        sample_weigths = labeling_info['t_value'].reindex(X_train.index).abs()


    ### CROS VALIDATION STEPS
    if cv_type == 'purged_kfold':
        cv = ml.cross_validation.PurgedKFold(
            n_splits=cv_number,
            samples_info_sets=labeling_info['t1'].reindex(X_train.index))


    ### MODEL WITH SKLEARN
    
    # estimator
    rf = RandomForestClassifier(criterion='entropy',
                                class_weight='balanced_subsample')
    
    # grid search
    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'n_estimators': [500, 1000],
        'max_features': [5, 10, 15, 20],
        'max_leaf_nodes': [4, 8, 16, 32]
        }
    tune_search = TuneGridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        early_stopping=False,
        scoring='f1',
        n_jobs=16,
        cv=cv,
        verbose=1
    )
    tune_search.fit(X_train, y_train, sample_weight=sample_weigths)
    clf_predictions = tune_search.predict(X_test)
    tune_search.cv_results_
    tune_search.best_params_  #max_depth 3, n_estimators 1000, max_features 10, max_leaf_nodes 4
    
    
    # random search
    param_random = {
        "n_estimators": randint(50, 1000),
        "max_depth": randint(2, 3),
        'max_features': randint(5, 25),
        'min_weight_fraction_leaf': randint(0.03, 0.1)
    }
    tune_search = TuneGridSearchCV(
        estimator=rf,
        param_grid=param_random,
        early_stopping=False,
        n_iter=5,
        scoring='f1',
        n_jobs=16,
        cv=cv,
        verbose=1
    )
    tune_search.fit(X_train, y_train, sample_weight=sample_weigths)
    clf_predictions = tune_search.predict(X_test)
    tune_search.cv_results_
    tune_search.best_index_
    tune_search.best_params_

    
    ####### TRY TUNE-SKELARN WHEN THEY ANSWER ON MY ISSUES QUESTION #######
    param_bayes = {
        "n_estimators": (50, 1000),
        "max_depth": (2, 7),
        'max_features': (1, 30)
        # 'min_weight_fraction_leaf': (0.03, 0.1, 'uniform')
    }
    tune_search = TuneSearchCV(
        rf,
        param_bayes,
        search_optimization='bayesian',
        max_iters=10,
        scoring='f1',
        n_jobs=16,
        cv=cv,
        verbose=1
    )
    tune_search.fit(X_train, y_train, sample_weight=sample_weigths)
    clf_predictions = tune_search.predict(X_test)
    tune_search.cv_results_
    tune_search.best_index_
    tune_search.best_params_
    ####### TRY TUNE-SKELARN WHEN THEY ANSWER ON MY ISSUES QUESTION #######

    
    # clf = GridSearchCV(rf,
    #                 param_grid=parameters,
    #                 scoring='f1',
    #                 n_jobs=16,
    #                 cv=cv)
    # clf.fit(X_train, y_train, sample_weight=sample_weigths)
    # max_depth, n_features, max_leaf_nodes, n_estimators = clf.best_params_.values()
    
    # model scores
    # clf_predictions = clf.predict(X_test)
    clf_f1_score = sklearn.metrics.f1_score(y_test, clf_predictions)
    clf_accuracy_score = sklearn.metrics.accuracy_score(y_test, clf_predictions)
    print(f'f1_score: {clf_f1_score}')
    print(f'f1_score: {clf_f1_score}')
    print(f'optimal_max_depth: {max_depth}')
    print(f'optimal_n_features: {n_features}')
    print(f'optimal_max_leaf_nodes {max_leaf_nodes}')
    print(f'optimal_n_estimators {n_estimators}')
    save_id = f'{max_depth}{n_features}{max_leaf_nodes}{n_estimators}{str(clf_f1_score)[2:6]}'


    # retrain the model if mean score is high enough (higher than 0.5)
    if clf_f1_score < 0.55:
        print('good_performance: False')
    else:
        print('good_performance: True')
        # refit best model and show results
        rf_best = RandomForestClassifier(criterion='entropy',
                                        max_features=n_features,
                                        min_weight_fraction_leaf=0.05,
                                        max_depth=max_depth,
                                        n_estimators=n_estimators,
                                        max_leaf_nodes=max_leaf_nodes,
                                        class_weight='balanced',
                                        n_jobs=16)
        rf_best.fit(X_train, y_train, sample_weight=sample_weigths)

        ### CLF METRICS
        tml.modeling.metrics_summary.clf_metrics(
            rf_best, X_train, X_test, y_train, y_test, avg='binary')  # HAVE TO FIX
        # tml.modeling.metrics_summary.plot_roc_curve(
        # clf, X_train, X_test, y_train, y_test, name='rf_')


        # ### FEATURE SELECTION        
        def feature_importance_values(clf, X_train, y_train, plot_name):

            # clone clf to not change it
            clf_ = sklearn.clone(clf)
            clf_.fit(X_train, y_train)

            # SHAP values
            explainer = shap.TreeExplainer(model=clf_, model_output='raw')
            shap_values = explainer.shap_values(X_train)
            vals= np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame(
                list(zip(X_train.columns, sum(vals))),
                columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(
                by=['feature_importance_vals'], ascending=False, inplace=True)
            
            # save plots
            # create directory if it does not exists
            if not os.path.exists('plots'):
                os.makedirs('plots')
            saving_path = Path(f'plots/shap_{plot_name}.png')
                    
            # shap plot
            shap.summary_plot(shap_values, X_train,
                            plot_type='bar', max_display=15,
                            show=False)
            plt.savefig(saving_path)
            
            # random forest default feature importance
            importances = pd.Series(clf_.feature_importances_, index=X_train.columns)
            importances = importances.sort_values(ascending=False)
            
            # mean decreasing impurity
            mdi_feature_imp = ml.feature_importance.mean_decrease_impurity(
                clf_, X_train.columns)
            
            # mean decreasing accuracy (COMMENT DUE TO EFFICIENCY PROBLEM)
            # mda_feature_imp = ml.feature_importance.mean_decrease_accuracy(
            #     clf_, X_train, y_train, cv, scoring=log_loss,
            #     sample_weight_train=sample_weigths.values)

            # base estimator for mlfinlab features importance (COMMENT DUE TO EFFICIENCY PROBLEM)
            # rf_best_one_features = RandomForestClassifier(
            #     criterion='entropy',
            #     max_features=1,
            #     min_weight_fraction_leaf=0.05,
            #     max_depth=max_depth,
            #     n_estimators=n_estimators,
            #     max_leaf_nodes=max_leaf_nodes,
            #     class_weight='balanced',
            #     n_jobs=16)
            
            # # doesn't work?
            # sfi_feature_imp = ml.feature_importance.single_feature_importance(
            #     rf_best_one_features, X_train, y_train, cv,
            #     scoring=sklearn.metrics.accuracy_score,
            #     sample_weight_train=sample_weigths.values)
            
            return feature_importance, importances, mdi_feature_imp


        def save_files(objects, file_names, directory='important_features'):            
            # create directory if it does not exists
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # save files to directory
            for df, file_name in zip(objects, file_names):
                saving_path = Path(f'{directory}/{file_name}')
                if ".csv" not in file_names: 
                    df.to_csv(saving_path)


        # save feature importance tables and plots
        shap_values, importances, mdi_feature_imp = feature_importance_values(
            rf_best, X_train, y_train, plot_name=save_id)
        save_files([shap_values, importances, mdi_feature_imp],
                   file_names=[f'shap_{save_id}.csv',
                               f'rf_importance_{save_id}.csv',
                               f'mpi_{save_id}.csv'],
                   directory='important_features')
        
        
        # fival = tml.modeling.feature_importance.feature_importance_values(
        #     rf_best, X_train, y_train)
        # fivec = tml.modeling.feature_importance.feature_importnace_vec(
        #     fival, X_train)
        # tml.modeling.feature_importance.plot_feature_importance(fival, X_train, name='rf_')

        
        # ### REFIT THE MODEL WITH MOST IMPORTANT FEATURES
        fi_cols = shap_values['col_name'].head(keep_important_features)
        X_train_important = X_train[fi_cols]
        X_test_important = X_test[fi_cols]
        clf_important = rf_best.fit(X_train_important, y_train)
        tml.modeling.metrics_summary.clf_metrics(
            clf_important, X_train_important,
            X_test_important, y_train, y_test, avg='binary', prefix='fi_')


        ### BACKTESTING (RADI)

        # BUY-SELL BACKTESTING STRATEGY
        # true close 
        time_range = pd.date_range(X_test.index[0], X_test.index[-1], freq='1Min')
        close = data.close_orig.reindex(time_range).to_frame().dropna()
        # predictions on test set
        predictions = pd.Series(rf_best.predict(X_test_important), index=X_test_important.index)
        # plot cumulative returns
        hold_cash = tml.modeling.backtest.hold_cash_backtest(close, predictions)
        # fig = hold_cash[['close_orig', 'cum_return']].plot().get_figure()
        # fig.savefig(f'backtest_hold_cash.png')

        # # VECTORBT
        # positions = pd.concat([close, predictions.rename('position')], axis=1)
        # positions = tml.modeling.backtest.enter_positions(positions.values)
        # positions = pd.DataFrame(positions, index=close.index, columns=['close', 'position'])
        # entries = (positions[['position']] == 1).vbt.signals.first()  # buy at first 1
        # exits = (positions[['position']] == -1).vbt.signals.first()  # sell at first 0
        # portfolio = vbt.Portfolio.from_signals(close, entries, exits,
        #                                     slippage=vectorbt_slippage,
        #                                     fees=vectorbt_fees)
        # print(f'vectorbt_total_return: {portfolio.total_return}')

        # #TRIPLE-BARRIER BACKTEST
        # tbpred = labeling_info.loc[predictions.index]
        # tbpred['ret_adj'] = np.where(tbpred['bin']==predictions, np.abs(tbpred['ret']), -np.abs(tbpred['ret']))
        # total_return = (1 + tbpred['ret_adj']).cumprod().iloc[-1]
        # print(f'tb_return_nofees_noslippage: {total_return}')





        ### SAVE THE MODEL AND FEATURES
        # joblib.dump(clf, "rf_model_25_ts.pkl")
        # pd.Series(X_train_important.columns).to_csv('feature_names_25_ts.csv', sep=',')
        # serialized_model = tml.modeling.utils.serialize_random_forest(clf)
        # with open('rf_model_25_ts.json', 'w') as f:
        #     json.dump(serialized_model, f)


        ### BACKTEST STATISTICS 

        # def pyfolio_sheet(returns):
        #     daily_returns = returns.resample('D').mean().dropna()
        #     perf_func = pf.timeseries.perf_stats
        #     perf_stats_all = perf_func(returns=daily_returns, 
        #                                factor_returns=None)
        #     return perf_stats_all

        # strategy_pf = pyfolio_sheet(hold_cash['return'])
        # bencha_pf = pyfolio_sheet(data.close_orig.resample('D').last().
        #                             dropna().pct_change())
        # pf_sheet = pd.concat([bencha_pf.rename('banchmark'),
        #                       strategy_pf.rename('strategy')], axis=1)




        # import  mlfinlab.backtest_statistics as bs
        # def backtest_stat(returns):
        #     # RUNS
        #     pos_concentr, neg_concentr, hour_concentr = bs.all_bets_concentration(returns, frequency='min')
        #     drawdown, tuw = bs.drawdown_and_time_under_water(returns, dollars = False)
        #     drawdown_dollars, _ = bs.drawdown_and_time_under_water(returns, dollars = True)

        #     # EFFICIENCY
        #     days_observed = (price_series.index[-1] - price_series.index[0]) / np.timedelta64(1, 'D')
        #     cumulated_return = price_series[-1]/price_series[0]
        #     annual_return = (cumulated_return)**(365/days_observed) - 1
        #     print('Annualized average return from the portfolio is' , annual_return)

        #     # merge all statistics to dictionary
        #     backtest_statistics = {
        #         'Positive concetration': pos_concentr,
        #         'Negative concetration': neg_concentr,
        #         'Hour concetration': hour_concentr,
        #         'The 95th percentile Drawdown': drawdown.quantile(.95),
        #         'The 95th percentile Drawdown in dollars': drawdown_dollars.quantile(.95),
        #         'The 95th percentile of Time under water': tuw.quantile(.95),
        #         'Maximum Drawdown': drawdown.max(),
        #         'Maximum Drawdown in dolars': drawdown_dollars.max(),
        #         'Maximum Drawdown time': tuw.max()
        #     }
        #     # dictionary to dataframe    
        #     df = pd.DataFrame.from_dict(backtest_statistics, orient='index')

        #     return df


        # returns = hold_cash['return'].dropna()
        # price_series = hold_cash['adjusted_close'].dropna()
        # backtest_stat(returns)
