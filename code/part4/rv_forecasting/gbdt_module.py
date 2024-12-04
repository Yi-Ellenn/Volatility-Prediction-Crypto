import os
import datetime
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

from loguru import logger
from pathlib import Path
from dateutil.relativedelta import relativedelta

from typing import Dict
from rv_forecasting import *

Cache_path = '/Users/zehoo/Documents/Course/波动率预测专题课程/code/Lecture_4/ModelCache'

class RVForecasting_GBDT:
    def __init__(self, horizon: str, lag:int, root_dir:str=Cache_path, percentage_error:bool = True):
        self.root_dir = Path(root_dir)
        self.horizon = horizon
        self.lag = lag
        self.model = None
        self.model_folder = os.path.join(root_dir, f'RV_{self.horizon}_Models')
        self.model_prefix = "RVLight"
        self.percentage_error = percentage_error
    
    def get_model_path(self, model_cutoff:datetime.datetime)->str:
        model_name = self.get_model_name(model_cutoff)
        model_path = os.path.join(self.model_folder, model_name)
        return model_path
    
    def get_model_name(self, model_cutoff:datetime.datetime)->str:
        model_name = model_cutoff.strftime("%Y%m%d%H%M%S")
        if self.model_prefix:
            model_name = self.model_prefix + "_" + model_name
        return model_name
    
    def is_model_existed(self, model_cutoff:datetime.datetime)->bool:
        model_path = self.get_model_path(model_cutoff)
        return os.path.exists(model_path)
    
    def load_model(self, model_cutoff)->lgb.Booster:
        model_path = self.get_model_path(model_cutoff)
        with open(model_path, 'rb') as f:
            model_str = pickle.load(f)
        model = lgb.Booster(model_str = model_str)
        return model

    def save_model(self, model:lgb.Booster, model_cutoff:datetime)->None:
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        model_path = self.get_model_path(model_cutoff)
        model_str = model.model_to_string()
        with open(model_path, "wb") as f:
            pickle.dump(model_str, f)
    
    def FindModel(self, date:datetime.datetime):
        pred_start = (date.normalize() - pd.offsets.MonthEnd(1) + pd.Timedelta(days=self.lag)).strftime('%Y%m%d')
        pred_end = (date.normalize() + pd.offsets.MonthEnd(0) + pd.Timedelta(days=self.lag)).strftime('%Y%m%d')
        model_name = f'RV_{self.horizon}_{pred_start}_{pred_end}'
        querry = self.model_path.glob(f'{model_name}.txt')
        try:
            next(querry)
        except StopIteration:
            logger.error(f"No model called {model_name}")
            return False
        else:
            return True

    def LoadModel(self, date:datetime.datetime):
        pred_start = (date.normalize() - pd.offsets.MonthEnd(1) + pd.Timedelta(days=self.lag)).strftime('%Y%m%d')
        pred_end = (date.normalize() + pd.offsets.MonthEnd(0) + pd.Timedelta(days=self.lag)).strftime('%Y%m%d')
        model_name = f'RV_{self.horizon}_{pred_start}_{pred_end}'
        model = lgb.Booster(model_file=self.model_path.joinpath(f'{model_name}.txt'))
        self.model = model
    
    def train(self, X:np.array, y:np.array, params:Dict):
        train_weights = 1 / (np.square(y) + 0.001) if self.percentage_error else [1.0] * len(y)
        train_dataset = lgb.Dataset(X, y, weight = train_weights, categorical_feature=['base'])
        lgb_params = params.copy()
        lgb_params.pop("n_iters")
        model = lgb.train(
            params = lgb_params,
            num_boost_round = params['n_iters'],
            train_set = train_dataset,
            valid_sets = [train_dataset],
            callbacks = [lgb.log_evaluation(250)],
            categorical_feature = ["base"]
        )
        return model

    def predict(self, data:np.array, model:lgb.Booster):
        res = model.predict(data)
        res = pd.DataFrame(np.vstack([res, data["base"].values]).T, columns=['predictions', 'base'], index = data.index)
        res = res.set_index("base", append = True)
        res = res["predictions"].unstack()
        res.columns = [RE_BASE_ENCODING[int(x)] for x in res.columns]
        return res 