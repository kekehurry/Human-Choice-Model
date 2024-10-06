from .utils.data_preparation import prepare_train_data, prepare_test_data
from .utils.neo4j_initialization import prepare_neo4j
from .utils.cypher_generation import get_llm_cypher, get_cypher_query
from .utils.similarity_score import get_similarity_score, get_similar_nodes
from .utils.link_prediction import create_behavior_subgraph, get_recommendation
from .utils.llm_choice import get_llm_choice
from .utils.llm_choice_without_context import get_llm_choice_without_context
from .utils.init_settings import init_settings
from .utils.evaluation import read_log_data, evaluate_with_cv
import pandas as pd
import numpy as np
import random
import json
import os


class ChoiceModel:
    def __init__(self, data_dir='../data', desire='Eat', choice_type='mode', sample_num=1000, skip_init=False, seed=42, skip_test=False):
        self.data_dir = data_dir
        self.choice_type = choice_type
        self.log_dir = os.path.join(
            data_dir, f'logs/{sample_num}')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.desire = desire
        self.sample_num = sample_num
        self.logs = pd.DataFrame(
            columns=['person_id', 'profile', 'top_k', 'desire', 'city', 'cypher',
                     'amenity_recommendation', 'amenity_llm_choice', 'amenity_final_choice',
                     'mode_recommendation', 'mode_llm_choice', 'mode_final_choice',])
        if not skip_init:
            self.train_data_path = self._prepare_train_data(
                sample_num=sample_num, desire=desire)
            if not skip_test:
                self.test_data_path = self._prepare_test_data(desire=desire)
            self._prepare_neo4j(self.train_data_path)
        else:
            self.train_data_path = os.path.join(
                self.data_dir, f'train/{sample_num}/{desire}')

        if not skip_test:
            self.test_data_path = os.path.join(
                self.data_dir, f'test/{desire}.csv')
            self.log_data_path = os.path.join(
                self.log_dir, f'{self.desire}.csv')
        self._set_seed(seed)
        return

    def _set_seed(self, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        return random_seed

    def _prepare_train_data(self, sample_num, desire):
        return prepare_train_data(self.data_dir, sample_num, desire)

    def _prepare_test_data(self, desire):
        return prepare_test_data(self.data_dir, desire)

    def _prepare_neo4j(self, train_dir):
        return prepare_neo4j(train_dir)

    def _get_llm_cypher(self, profile, top_k, desire):
        return get_llm_cypher(profile, top_k, desire)

    def _get_cypher_query(self, cypher):
        return get_cypher_query(cypher)

    def _get_similar_score(self, profile, query_results):
        return get_similarity_score(profile, query_results)

    def _get_similar_nodes(self, profile, query_results, k):
        return get_similar_nodes(profile, query_results, k)

    def _create_behavior_subgraph(self, query_results):
        return create_behavior_subgraph(query_results)

    def _get_recommendation(self, G, choice_type):
        return get_recommendation(G, choice_type)

    def _get_llm_choice(self, profile, desire, choice_type, recommendation, city):
        return get_llm_choice(profile, desire, choice_type, recommendation, city)

    def _get_llm_choice_without_context(self, profile, desire, choice_type, options, city):
        return get_llm_choice_without_context(profile, desire, choice_type, options, city)

    def _get_final_choice(self, llm_choice):
        try:
            llm_choice = json.loads(llm_choice)
            options = [item['choice'].strip()
                       for item in llm_choice['final_answer']]
            weights = [item['weight']
                       for item in llm_choice['final_answer']]
            choice = random.choices(options, weights=weights, k=1)[0]
        except Exception as e:
            return None
        return choice

    def _read_log_data(self, log_path, test_path):
        return read_log_data(log_path, test_path)

    def _evaluate_with_cv(self, data, choice_type, figsize=(20, 5), plot=False):
        return evaluate_with_cv(data, choice_type, figsize=figsize, plot=plot)

    def infer(self, profile, top_k=50, city='Boston', mode='infer'):
        desire = self.desire

        # mannally extract cypher information for more accurate results in experiment
        if mode == 'experiment':
            person_id = profile['person_id']
            cypher = profile['cypher']
            profile = f"A {profile['age']} year old person, living in a {profile['family_structure']} family with {profile['household_size']} members. The person has {profile['vehicles']} vehicles and an annual income of {profile['individual_income']} dollars."
        else:
            person_id = random.randint(0, 10000)
            profile = str(profile)
            cypher = self._get_llm_cypher(profile, top_k, desire)

        query_results = self._get_cypher_query(cypher)
        query_results = self._get_similar_score(profile, query_results)
        behavior_graph = self._create_behavior_subgraph(query_results)
        amenity_recommendation = self._get_recommendation(
            behavior_graph, choice_type='Amenity')
        amenity_llm_choice = self._get_llm_choice(
            profile, desire, 'Amenity', amenity_recommendation, city)
        amenity_final_choice = self._get_final_choice(amenity_llm_choice)
        mode_recommendation = self._get_recommendation(
            behavior_graph, choice_type='Mode')
        mode_llm_choice = self._get_llm_choice(
            profile, desire, 'Mode', mode_recommendation, city)
        mode_final_choice = self._get_final_choice(mode_llm_choice)
        self.logs.loc[len(self.logs)] = [person_id, profile, top_k, desire, city, cypher,
                                         amenity_recommendation, amenity_llm_choice, amenity_final_choice,
                                         mode_recommendation, mode_llm_choice, mode_final_choice]
        return amenity_final_choice, mode_final_choice

    def infer_without_context(self, profile, city='Boston', mode='infer'):
        desire = self.desire
        if mode == 'experiment':
            person_id = profile['person_id']
            profile = f"A {profile['age']} year old person, living in a {profile['family_structure']} family with {profile['household_size']} members. The person has {profile['vehicles']} vehicles and an annual income of {profile['individual_income']} dollars."
        else:
            person_id = random.randint(0, 10000)
            profile = str(profile)
        amenity_options = ['F&B Eatery/Drinking Places', 'F&B Eatery/Snack and Nonalcoholic Beverage Bars',
                           'F&B Eatery/Full-Service Restaurants', 'F&B Eatery/Limited-Service Restaurants',
                           'F&B Eatery/Special Food Services']
        mode_options = ['Walking', 'Car', 'Public_transport']

        amenity_llm_choice = self._get_llm_choice_without_context(
            profile, desire, 'Amenity', amenity_options, city)
        amenity_final_choice = self._get_final_choice(amenity_llm_choice)
        mode_llm_choice = self._get_llm_choice_without_context(
            profile, desire, 'Mode', mode_options, city)
        mode_final_choice = self._get_final_choice(mode_llm_choice)
        self.logs.loc[len(self.logs)] = [person_id, profile, 0, desire, city, '',
                                         '', amenity_llm_choice, amenity_final_choice,
                                         '', mode_llm_choice, mode_final_choice]
        return amenity_final_choice, mode_final_choice

    def save_logs(self, log_data_path=None):
        if log_data_path:
            self.log_data_path = log_data_path
        self.logs.to_csv(self.log_data_path, index=False)
        return

    def evaluate(self, figsize=(20, 5), plot=False):
        log_path = self.log_data_path
        test_path = self.test_data_path
        choice_type = self.choice_type
        data = self._read_log_data(log_path, test_path)
        loss, error, kl_divergence = self._evaluate_with_cv(
            data, choice_type, figsize=figsize, plot=plot)
        return loss, error, kl_divergence

    def visualize_graph(self):
        return
