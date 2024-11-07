from sklearn.neural_network import MLPClassifier
from .base_model import BaseModel


class MLP(BaseModel):
    def __init__(self, data_dir, desire='Eat', choice_type='mode', sample_num=1000, seed=42):
        super().__init__(data_dir, desire, choice_type, sample_num, seed)
        self.model = MLPClassifier(max_iter=20000, random_state=self.seed)
        self.param_grid = {
            'hidden_layer_sizes': [(50, 50), (100, 100), (150, 150)],
            'activation': ['tanh', 'relu'],
            'alpha': [1e-5, 1e-4, 1e-3],
            'learning_rate_init': [1e-4, 1e-3, 1e-2],
        }

    def init(self, max_iter=20000):
        model = self.model.__class__(
            max_iter=max_iter, random_state=self.seed)
        return model
