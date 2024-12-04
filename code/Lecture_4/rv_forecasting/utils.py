import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from loguru import logger
import matplotlib.pyplot as plt


def load_data(start_date, end_date, base, path):
    """_summary_

    Args:
        start_date (_type_): 起始日期
        end_date (_type_): 结束日期
        base (_type_): 标的
        path (_type_): 数据路径

    Returns:
        _type_: DataFrame
    """
    df = pd.read_parquet(path) 
    assert 'date' in df.columns
    df['date'] = pd.to_datetime(df['date'])
    df_clip = df[(df['date']>=start_date) & (df['date'] <= end_date)]
    df_base = df_clip[df['base'] == base]
    return df_base

def smoothing_Z_score(df:pd.DataFrame, sz:int, tr:int):
    df_copy = df.copy()
    for column in df.columns:
        # 计算滚动窗口内的均值和标准差
        df_copy[column + '_mean'] = df_copy[column].rolling(window=sz, min_periods=1).mean()
        df_copy[column + '_std'] = df_copy[column].rolling(window=sz, min_periods=1).std()
        
        # 计算异常值的阈值范围
        df_copy[column + '_lower_bound'] = df_copy[column + '_mean'] - tr * df_copy[column + '_std']
        df_copy[column + '_upper_bound'] = df_copy[column + '_mean'] + tr * df_copy[column + '_std']
        
        # 标记异常值
        outliers = (df_copy[column] < df_copy[column + '_lower_bound']) | (df_copy[column] > df_copy[column + '_upper_bound'])
        
        # 裁剪异常值
        df_copy.loc[outliers, column] = np.clip(df_copy.loc[outliers, column], 
                                                df_copy[column + '_mean'] - tr * df_copy[column + '_std'], 
                                                df_copy[column + '_mean'] + tr * df_copy[column + '_std'])   
    # 删除中间结果列
    df_copy = df_copy.drop(columns=[col for col in df_copy.columns if col.endswith(('_mean', '_std', '_lower_bound', '_upper_bound') )])
    return df_copy

def get_features_importance(models, importance_type: str='gain', fig_path:str = '/Users/zehoo/Documents/Course/波动率预测专题课程/code/Lecture_4/Cache'):
    # importance type: gain, split
    if not all(isinstance(model, lgb.Booster) for model in models):
        logger.error('Invalid model type')
    importance_table = {}
    for idx, model in enumerate(models):
        importance = model.feature_importance(importance_type=importance_type)
        importance_table[idx] = pd.Series(importance, index = model.feature_name())
    importance_table = pd.DataFrame.from_dict(importance_table, orient = "index")
    mean_importance = importance_table.mean()
    std_importance = importance_table.std()
    result = pd.DataFrame()
    result["mean"] = mean_importance
    result["std"] = std_importance
    result["multiplier"] = mean_importance / std_importance
    result = result.sort_values(by = "mean", ascending = True)
    print(result)
    fig, ax = plt.subplots(figsize=(16, 10))
    labels = result.index
    y_pos = range(len(result))
    importance_mean = result['mean']
    ax.barh(y_pos, importance_mean)
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel('importance')
    ax.set_ylabel('features')
    plt.tight_layout()
    plt.savefig(f'{fig_path}/features_importance.jpg')
    
def cal_r2(dataset, predictions):
    target = dataset[['base', 'rv_24']]
    obs_res = pd.concat([target[target['base']==0]['rv_24'], target[target['base']==1]['rv_24']], axis=1)
    obs_res.columns = ['BTC', 'ETH']
    mask = obs_res.isna()
    return r2_score(predictions[~mask].dropna(), obs_res[~mask].dropna())

def cal_corr(dataset, predictions):
    target = dataset[['base', 'rv_24']]
    obs_res = pd.concat([target[target['base']==0]['rv_24'], target[target['base']==1]['rv_24']], axis=1)
    obs_res.columns = ['BTC', 'ETH']
    mask = obs_res.isna()
    return predictions[~mask].dropna().corrwith(obs_res[~mask].dropna()).mean()
    
    
    