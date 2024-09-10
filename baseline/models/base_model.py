import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold
from .data import get_dataset, get_data
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random


class BaseModel(BaseEstimator, ClassifierMixin):
    def __init__(self, data_dir, desire='Eat', choice_type='mode', sample_num=1000, seed=42):
        self.data_dir = data_dir
        self.desire = desire
        self.sample_num = sample_num
        self.result_dir = os.path.join(self.data_dir, "results")
        self.experiment_dir = os.path.join(self.data_dir, "experiments")
        for dir in [self.result_dir, self.experiment_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)
        self.seed = self.set_seed(seed)
        self.model = None
        self.param_grid = None
        self.best_params = None
        self.choice_type = choice_type
        return

    def set_seed(self, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        return random_seed

    def load_dataset(self):
        self.train_file = os.path.join(
            self.data_dir, f"train/{self.desire}.csv")
        self.test_file = os.path.join(self.data_dir, f"test/{self.desire}.csv")

        train_dataset, test_dataset, mapping = get_dataset(
            self.train_file, self.test_file, random_state=self.seed)
        train_dataset = train_dataset[:self.sample_num].reset_index(drop=True)
        test_dataset = test_dataset.reset_index(
            drop=True)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.mapping = mapping

        return train_dataset, test_dataset, mapping

    def prepare_data(self):
        X_train, y_train = get_data(
            self.train_dataset, self.mapping, y_prop=f"target_{self.choice_type}")
        X_test, y_test = get_data(
            self.test_dataset, self.mapping, y_prop=f"target_{self.choice_type}")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.drop('person_id', axis=1))
        y_train = np.array(y_train.drop('person_id', axis=1)).squeeze()
        X_test = scaler.transform(X_test.drop('person_id', axis=1))
        y_test = np.array(y_test.drop('person_id', axis=1)).squeeze()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        return X_train, y_train, X_test, y_test

    def optimize(self):
        self.init()
        class_labels = range(
            len(self.mapping[f'target_{self.choice_type}'].keys()))
        scorer = make_scorer(log_loss, greater_is_better=False,
                             needs_proba=True, labels=class_labels)
        grid_search = GridSearchCV(
            self.model, self.param_grid, cv=5, scoring=scorer, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        self.best_params = best_params

        return best_params, best_score

    def init(self):
        self.model = self.model.__class__(random_state=self.seed)

    def train(self):
        # init model
        self.init()
        # set best params
        if self.best_params:
            self.model.set_params(**self.best_params)
        # train
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        # probs = self.predict_proba(X_test)
        return self.model.predict(X_test)
        # return np.array([np.random.choice(row.shape[0], p=row) for row in probs])

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def get_results(self,):

        results = self.test_dataset.copy()

        y_pred = self.predict(self.X_test)
        y_pred_proba = self.predict_proba(self.X_test)

        pred_map = self.mapping[f'target_{self.choice_type}']
        reversed_map = {v: k for k, v in pred_map.items()}

        results[f'predict_{self.choice_type}_numeric'] = y_pred
        results[f'predict_{self.choice_type}'] = results[f'predict_{self.choice_type}_numeric'].map(
            reversed_map)
        results[f'{self.choice_type}_probs'] = [y_pred_proba[i]
                                                for i in range(y_pred_proba.shape[0])]

        results[f'target_{self.choice_type}_numeric'] = results[f'target_{self.choice_type}'].map(
            pred_map)

        return results

    def _cal_kl_devergence(self, P, Q):
        epsilon = 1e-10

        P = np.array(P)
        P = P / np.sum(P)
        P = np.clip(P, epsilon, 1)

        Q = np.array(Q)
        Q = Q / np.sum(Q)
        Q = np.clip(Q, epsilon, 1)

        # Calculate the KL divergence
        KL_divergence = np.sum(P * np.log(P / Q))
        return KL_divergence

    def get_error(self, data, x, figsize=(20, 5), plot=False):

        y = f'target_{self.choice_type}'
        pred_y = f'predict_{self.choice_type}'
        x_order = list(self.mapping[x].keys())
        y_order = list(self.mapping[y].keys())

        contingency_table = pd.crosstab(data[y], data[x])
        percentage_table = contingency_table.div(
            contingency_table.sum(axis=0), axis=1)
        percentage_table = percentage_table.reindex(
            x_order, axis=1, fill_value=0)
        percentage_table = percentage_table.reindex(
            y_order, axis=0, fill_value=0)

        predict_contingency_table = pd.crosstab(
            data[pred_y], data[x])
        predict_percentage_table = predict_contingency_table.div(
            contingency_table.sum(axis=0), axis=1)
        predict_percentage_table = predict_percentage_table.reindex(
            x_order, axis=1, fill_value=0)
        predict_percentage_table = predict_percentage_table.reindex(
            y_order, axis=0, fill_value=0)

        error_table = predict_percentage_table-percentage_table

        average_error = error_table.abs().mean()
        total_average_error = average_error.mean()

        kl_divergence = self._cal_kl_devergence(
            percentage_table, predict_percentage_table)

        if plot:
            # draw heatmap
            fig, axs = plt.subplots(1, 3, figsize=figsize,
                                    sharex=True, sharey=True)

            ax1 = sns.heatmap(percentage_table, annot=True, fmt=".2f",
                              cmap="YlGnBu", vmin=0, vmax=1, ax=axs[0])
            ax2 = sns.heatmap(predict_percentage_table, annot=True,
                              fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=axs[1])
            ax3 = sns.heatmap(error_table, annot=True, fmt=".2f",
                              cmap="vlag", vmin=-0.5, vmax=0.5, ax=axs[2])

            ax1.set_title("Ground Truth")
            ax2.set_title("Prediction")
            ax3.set_title("Percentage Error")
            for ax in axs:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.show()
        return total_average_error, kl_divergence

    def evaluate_with_cv(self, figsize=(20, 5), plot=False):
        losses = []

        data = self.get_results()

        y_true = np.array(data[f'target_{self.choice_type}'])

        y_true_numeric = np.array(data[f'target_{self.choice_type}_numeric'])
        y_pred_probs = np.vstack(data[f'{self.choice_type}_probs'])

        # split X_test and y_test into cv folds
        num_classes = len(self.mapping[f'target_{self.choice_type}'].keys())
        categories = [list(range(num_classes))]
        encoder = OneHotEncoder(sparse_output=False, categories=categories)
        y_onehot = encoder.fit_transform(y_true_numeric.reshape(-1, 1))

        indexes = [np.arange(len(y_true))]
        for test_index in indexes:
            # log loss
            y_fold_onehot = y_onehot[test_index]
            y_fold_prob = y_pred_probs[test_index]
            loss = log_loss(y_fold_onehot, y_fold_prob)
            losses.append(loss)
        loss = np.mean(losses)
        # error
        error = {}
        kl_divergence = {}

        for x_label in ['age_group', 'income_group', 'household_size', 'vehicles', 'family_structure']:
            error[x_label], kl_divergence[x_label] = self.get_error(
                data, x_label, figsize=figsize, plot=plot)
        mean_error = np.mean(list(error.values()))
        mean_kl_divergence = np.mean(list(kl_divergence.values()))

        error['mean'] = mean_error
        kl_divergence['mean'] = mean_kl_divergence

        return loss, error, kl_divergence

    def run_experiment(self, best_params=None):
        # load dataset
        self.load_dataset()
        # scaler
        self.prepare_data()
        # optimize
        if best_params is None:
            self.optimize()
        else:
            self.best_params = best_params
        # train
        self.train()

        # evaluate
        loss, error, kl_divergence = self.evaluate_with_cv()

        return loss, error, kl_divergence
