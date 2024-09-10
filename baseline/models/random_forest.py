from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from .base_model import BaseModel


class RandomForest(BaseModel):
    def __init__(self, data_dir, desire='Eat', choice_type='mode', sample_num=1000, seed=42):
        super().__init__(data_dir, desire, choice_type, sample_num, seed)
        self.model = RandomForestClassifier()
        self.param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
