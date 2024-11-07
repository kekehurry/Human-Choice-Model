import random
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from .data_preparation import map_amenity
import numpy as np
import random


def get_mapping(desire):

    if desire == 'Eat':
        amenity_mapping = {
            'Limited-Service Restaurants': 0,
            'Nonalcoholic Bars': 1,
            'Full-Service Restaurants': 2,
            'Drinking Places': 3,
        }
    elif desire == 'Recreation':
        amenity_mapping = {
            'Leisure & Wellness': 0,
            'Entertainment': 1,
            'Cultural': 2,
            'Hotel': 3,
        }
    elif desire == 'Shop':
        amenity_mapping = {
            'Consumer Goods': 0,
            'Grocery': 1,
            'Durable Goods': 2,
        }
    else:
        raise ValueError("Invalid desire")

    mapping = {
        "household_size": {'1_person': 1, '2_person': 2, '3_person': 3,
                           '4_person': 4, '5_person': 5, '6_person': 6, '7_plus_person': 7},
        "vehicles": {'0': 0, '1': 1, '2': 2, '3_plus': 3},
        "family_structure": {'living_alone': 0, 'nonfamily_single': 1,
                             'family_single': 2, 'married_couple': 3},
        "target_mode": {'Walking': 0, 'Public_transport': 1, 'Car': 2},
        "target_amenity": amenity_mapping,
        "income_group": {'Debt': 0, 'Low': 1, 'Moderate': 2,
                         'High': 3, 'Very High': 4, 'Ultra High': 5},
        "age_group": {'Young Adult': 0, 'Adult': 1,
                      'Middle Age': 2, 'Senior': 3, 'Elderly': 4}
    }
    return mapping


def normalize(x):
    total = np.sum(x)
    norm_x = np.divide(x, total, out=np.zeros_like(x), where=total != 0)
    return norm_x


def cal_kl_devergence(P, Q):
    epsilon = 1e-10
    P = np.array(P)
    P = P / np.sum(P)
    P = np.clip(P, epsilon, 1)

    Q = np.array(Q)
    Q_sum = np.sum(Q)
    if Q_sum == 0:
        return 1
    Q = Q / Q_sum
    Q = np.clip(Q, epsilon, 1)

    # Calculate the KL divergence
    KL_divergence = np.sum(P * np.log(P / Q))
    return KL_divergence


def get_mode_probs(row, mapping):
    mode_mapping = mapping['target_mode']
    mode_num = max(mode_mapping.values()) + 1
    try:
        mode_llm_choice = json.loads(row['mode_llm_choice'])
        mode_probs = np.zeros(mode_num)
        for item in mode_llm_choice['final_answer']:
            option = item['choice'].strip()
            if 'Walk' in option:
                index = mode_mapping['Walking']
            elif 'Public' in option:
                index = mode_mapping['Public_transport']
            elif 'Car' in option:
                index = mode_mapping['Car']
            else:
                continue
            mode_probs[index] = item['weight']
        mode_probs = normalize(mode_probs)
    except Exception as e:
        return None
    return mode_probs


def get_amenity_probs(row, mapping):
    amenity_mapping = mapping['target_amenity']
    amenity_num = max(amenity_mapping.values()) + 1
    try:
        amenity_llm_choice = json.loads(row['amenity_llm_choice'])
        amenity_probs = np.zeros(amenity_num)
        for item in amenity_llm_choice['final_answer']:
            option = item['choice'].strip()
            if option in amenity_mapping:
                index = amenity_mapping[option]
                amenity_probs[index] = item['weight']
        amenity_probs = normalize(amenity_probs)
    except Exception as e:
        return None
    return amenity_probs


def get_mode_choice(row, mapping):
    mode_mapping = mapping['target_mode']
    mode_probs = row['mode_probs']
    if mode_probs is None:
        return None
    # mode_choice = np.random.choice(mode_num, p=mode_probs)
    mode_choice = random.choices(
        range(mode_probs.shape[0]), weights=mode_probs, k=1)[0]
    choice_key = list(mode_mapping.keys())[mode_choice]
    final_choice = mode_mapping[choice_key]
    return final_choice


def get_amenity_choice(row, mapping):
    amenity_mapping = mapping['target_amenity']
    amenity_probs = row['amenity_probs']
    if amenity_probs is None:
        return None
    # amenity_choice = np.random.choice(amenity_num, p=amenity_probs)
    amenity_choice = random.choices(
        range(amenity_probs.shape[0]), weights=amenity_probs, k=1)[0]
    choice_key = list(amenity_mapping.keys())[amenity_choice]
    final_choice = amenity_mapping[choice_key]
    return final_choice


def read_log_data(log_path, test_path, desire, test_num=1000):
    mapping = get_mapping(desire)
    amenity_mapping = mapping['target_amenity']
    reversed_amenity_mapping = {v: k for k, v in amenity_mapping.items()}
    mode_mapping = mapping['target_mode']
    reversed_mode_mapping = {v: k for k, v in mode_mapping.items()}

    test_data = pd.read_csv(test_path, index_col=False)
    income_bins = [-float('inf'), 0, 30000, 60000, 90000, 120000, float('inf')]
    income_labels = ['Debt', 'Low', 'Moderate',
                     'High', 'Very High', 'Ultra High']
    age_bins = [-float('inf'), 18, 30, 40, 50, 60, float('inf')]
    age_labels = ['Teen', 'Young Adult', 'Adult',
                  'Middle Age', 'Senior', 'Elderly']

    test_data['income_group'] = pd.cut(
        test_data['individual_income'], bins=income_bins, labels=income_labels)
    test_data['age_group'] = pd.cut(
        test_data['age'], bins=age_bins, labels=age_labels)
    test_data['income_group'] = pd.cut(
        test_data['individual_income'], bins=income_bins, labels=income_labels)
    test_data['age_group'] = pd.cut(
        test_data['age'], bins=age_bins, labels=age_labels)
    test_data['target_mode'] = test_data['mode']
    test_data['target_amenity'] = test_data['target_amenity'].apply(
        lambda x: map_amenity(x, desire))

    # remove age_group=='Teen', because there is not enough data points in the train set
    test_data = test_data[test_data['age_group'] != 'Teen']

    log_data = pd.read_csv(log_path, index_col=False)
    log_data = log_data.drop(
        columns=['mode_recommendation', 'amenity_recommendation', 'cypher'])
    log_data['mode_probs'] = log_data.apply(
        get_mode_probs, axis=1, args=(mapping,))
    log_data['amenity_probs'] = log_data.apply(
        get_amenity_probs, axis=1,  args=(mapping,))
    log_data['predict_mode_numeric'] = log_data.apply(
        get_mode_choice, axis=1, args=(mapping,))
    log_data['predict_amenity_numeric'] = log_data.apply(
        get_amenity_choice, axis=1, args=(mapping,))
    log_data = log_data.dropna()

    pred_data = log_data[['person_id', 'mode_probs',
                          'amenity_probs', 'predict_mode_numeric', 'predict_amenity_numeric']]
    true_data = test_data[['person_id', 'income_group', 'household_size', 'vehicles', 'family_structure',
                           'age_group', 'target_amenity', 'target_mode']]

    merged_data = pd.merge(pred_data, true_data, on='person_id')

    merged_data['target_amenity_numeric'] = merged_data['target_amenity'].map(
        amenity_mapping)
    merged_data['target_amenity'] = merged_data['target_amenity_numeric'].map(
        reversed_amenity_mapping)

    merged_data['target_mode_numeric'] = merged_data['target_mode'].map(
        mode_mapping)
    merged_data['target_mode'] = merged_data['target_mode_numeric'].map(
        reversed_mode_mapping)

    merged_data['predict_mode'] = merged_data['predict_mode_numeric'].map(
        reversed_mode_mapping)
    merged_data['predict_amenity'] = merged_data['predict_amenity_numeric'].map(
        reversed_amenity_mapping)

    amenity_probs = np.vstack(merged_data['amenity_probs'])
    merged_data = merged_data[amenity_probs.sum(axis=1) == 1]

    mode_probs = np.vstack(merged_data['mode_probs'])
    merged_data = merged_data[mode_probs.sum(axis=1) == 1]

    merged_data = merged_data.dropna()
    merged_data = merged_data[:test_num]

    return merged_data


def get_error(data, x, choice_type, desire, figsize=(20, 5), plot=False):
    mapping = get_mapping(desire)

    y = f'target_{choice_type}'
    pred_y = f'predict_{choice_type}'

    # x_order = list(mapping[x].keys())
    # y_order = list(mapping[y].keys())
    x_order = list(mapping[x].keys())
    reversed_map = {v: k for k, v in mapping[y].items()}
    y_order = list(reversed_map.values())

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

    error_table = (predict_percentage_table-percentage_table)*100

    average_error = error_table.abs().mean()
    total_average_error = average_error.mean()

    kl_divergence = cal_kl_devergence(
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


def get_error_table(data, x_label, choice_type, desire):

    mapping = get_mapping(desire)

    y = f'target_{choice_type}'
    pred_y = f'predict_{choice_type}'
    # x_order = list(mapping[x_label].keys())
    # y_order = list(mapping[y].keys())
    x_order = list(mapping[x_label].keys())
    reversed_map = {v: k for k, v in mapping[y].items()}
    y_order = list(reversed_map.values())

    contingency_table = pd.crosstab(data[y], data[x_label])
    percentage_table = contingency_table.div(
        contingency_table.sum(axis=0), axis=1)
    percentage_table = percentage_table.reindex(
        x_order, axis=1, fill_value=0)
    percentage_table = percentage_table.reindex(
        y_order, axis=0, fill_value=0)

    predict_contingency_table = pd.crosstab(
        data[pred_y], data[x_label])
    predict_percentage_table = predict_contingency_table.div(
        contingency_table.sum(axis=0), axis=1)
    predict_percentage_table = predict_percentage_table.reindex(
        x_order, axis=1, fill_value=0)
    predict_percentage_table = predict_percentage_table.reindex(
        y_order, axis=0, fill_value=0)

    percentage_table = percentage_table*100
    predict_percentage_table = predict_percentage_table*100
    error_table = predict_percentage_table-percentage_table
    return percentage_table, predict_percentage_table, error_table


def evaluate_with_cv(data, choice_type, desire, figsize=(20, 5), plot=False):
    # error
    error = {}
    kl_divergence = {}

    for x_label in ['age_group', 'income_group', 'household_size', 'vehicles', 'family_structure']:
        error[x_label], kl_divergence[x_label] = get_error(
            data, x_label, choice_type, desire, figsize=figsize, plot=plot)
    mean_error = np.mean(list(error.values()))
    mean_kl_divergence = np.mean(list(kl_divergence.values()))
    error['mean'] = mean_error
    kl_divergence['mean'] = mean_kl_divergence

    return error, kl_divergence
