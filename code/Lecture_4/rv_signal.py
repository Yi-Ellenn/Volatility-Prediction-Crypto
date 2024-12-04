import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from rv_forecasting.data_module import DataModule, DataPreprocessing
from rv_forecasting.gbdt_module import RVForecasting_GBDT
from rv_forecasting.utils import get_features_importance, cal_r2, cal_corr
import warnings
warnings.filterwarnings("ignore")


# 预测目标，未来一天的波动率
HORIZONS = ["24h"]

# delay
TRAINING_DELAY_DAYS = 3

# 训练集长度
PERIODS = 720

# 预测标签
TARGET = 'rv_24'

# 数据路径
ROOT_DIR = '/Users/zehoo/Documents/Course/波动率预测专题课程/data/DC_Features_20190401_20221224.parquet'

# 基础特征
COL_NAMES = ['open', 'high', 'low', 'close', 'vwap', 'twap', 'awap', 'volume','tradeval', 'ntrade', 'ntrade_up', 'ntrade_down', 'ntrade_flat', 'close_ask', 'close_bid', 'close_asksize', 'close_bidsize', 'close_spread', 'nquote']

# 需要丢弃的特征
DROP_COLUMNS = ['open', 'high', 'low', 'close', 'vwap', 'twap', 'awap', 'volume',
    'tradeval', 'ntrade', 'ntrade_up', 'ntrade_down', 'ntrade_flat','close_ask', 'close_bid', 'close_asksize', 'close_bidsize']

# 模型参数
PARAMS = {
        'learning_rate': 0.05,  # 学习率
        'max_depth': 5, # 树的最大深度
        'lambda_l1': 0, # 正则项
        'lambda_l2': 0, # 正则项
        'objective': 'rmse', # 目标函数，希望rmse最小
        'boosting': 'gbdt', # 使用梯度提升树
        'verbosity': -1, # 是不是输出详细日志
        'n_jobs':-1,  # 并行线程数
        'n_iters': 500 # 迭代的次数
    }


class RVSignal:
    def __init__(self, train_size, horizon, importance_plot=False) -> None:
        self.importance_plot = importance_plot
        self.train_size = train_size
        self.horizon = horizon
        self.pred_module = RVForecasting_GBDT(self.horizon[0], TRAINING_DELAY_DAYS)
       
    def rolling_train_pred(self, dataset: pd.DataFrame):
        begin_time, end_time = dataset.index[0], dataset.index[-1]
        first_model_cutoff = begin_time.normalize() + pd.Timedelta(days=self.train_size)
        if first_model_cutoff.day != 1:
            first_model_cutoff += pd.offsets.MonthBegin()
        
        last_model_cutoff = end_time.normalize() - pd.Timedelta(days = TRAINING_DELAY_DAYS)
        if last_model_cutoff.day != 1:
            last_model_cutoff -= pd.offsets.MonthBegin()

        current_model_cutoff = first_model_cutoff
        predictions = []
        models = []
        while current_model_cutoff <= last_model_cutoff:
            train_data_start = current_model_cutoff - pd.Timedelta(days = self.train_size)
            train_data_end = current_model_cutoff - pd.Timedelta(days = 1) # due to 24h rv calculation
            train_data_mask = (dataset.index <= train_data_end) & (dataset.index > train_data_start)
            
            train_data = dataset.loc[train_data_mask]

            test_data_start = current_model_cutoff + pd.Timedelta(days = TRAINING_DELAY_DAYS)
            test_data_end = test_data_start + relativedelta(months = 1)
            test_data_mask = (dataset.index >= test_data_start) & (dataset.index < test_data_end)

            test_data = dataset.loc[test_data_mask]
            
            train_X, train_y = train_data.drop(TARGET, axis=1), train_data[TARGET].astype('float64')
            test_X = test_data.drop(TARGET, axis=1)

            train_X, train_y = train_X[np.isfinite(train_y).values].copy(), train_y[np.isfinite(train_y).values]

            if self.pred_module.is_model_existed(current_model_cutoff):
                model = self.pred_module.load_model(current_model_cutoff)
            else:
                model = self.pred_module.train(train_X, train_y, PARAMS)
                self.pred_module.save_model(model, current_model_cutoff)
                
            if current_model_cutoff == first_model_cutoff:
                front_data_mask = dataset.index < test_data_start
                front_data = dataset.loc[front_data_mask]
                front_X = front_data.drop(TARGET, axis=1)
                front_prediction = self.pred_module.predict(front_X, model)
                predictions.append(front_prediction)

            if len(test_X) > 0:
                prediction = self.pred_module.predict(test_X, model)
                predictions.append(prediction)
            
            current_model_cutoff += relativedelta(months = 1)
            models.append(model)
        
        predictions = pd.concat(predictions)
        if self.importance_plot:
            get_features_importance(models, importance_type='gain')
        return predictions
    
def main():
    freq = '1h'
    begin_time, end_time = '20190601', '20221201' # 开始时间，结束时间
    # BTC, ETH 两个品种定义DataModule
    dm = [DataModule(ROOT_DIR, freq, base) for base in ['BTC', 'ETH']]
    datasets = dict(zip(['BTC', 'ETH'], [dm_i.get_features(starttime=begin_time, endtime=end_time, columns=COL_NAMES) for dm_i in dm]))
    dp = DataPreprocessing(datasets=datasets, freq=freq)
    data = dp.data_collation(drop_columns=DROP_COLUMNS)
    rv_signal = RVSignal(PERIODS, HORIZONS, importance_plot=True)
    res = rv_signal.rolling_train_pred(data)
    res.to_csv('./rv_forecasting_result.csv')
    ic = cal_corr(data, res)
    print(f'Prediction IC: {ic}')
    return res
    
    

if __name__ == '__main__':
    main()

        
        
        
        
        


