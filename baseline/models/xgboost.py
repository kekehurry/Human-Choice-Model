import xgboost as xgb
from .base_model import BaseModel


class XGBoost(BaseModel):
    def __init__(self, data_dir, desire='Eat', choice_type='mode', sample_num=1000, seed=42):
        super().__init__(data_dir, desire, choice_type, sample_num, seed)
        self.model = xgb.XGBClassifier(objective='multi:softprob')
        self.param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'learning_rate': [1e-5, 1e-4, 1e-3],
            'subsample': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
        }
