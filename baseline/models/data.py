import numpy as np
import pandas as pd


def map_amenity(amenity, desire):
    if desire == 'Eat':
        if 'Limited-Service' in amenity:
            return 'Limited-Service Restaurants'
        elif 'Snack' in amenity:
            return 'Nonalcoholic Bars'
        elif 'Full-Service' in amenity:
            return 'Full-Service Restaurants'
        elif 'Drinking' in amenity:
            return 'Drinking Places'
        else:
            return 'Others'
    elif desire == 'Shop':
        if 'Consumer Goods' in amenity:
            return 'Consumer Goods'
        elif 'Grocery' in amenity:
            return 'Grocery'
        elif 'Durable Goods' in amenity:
            return 'Durable Goods'
        else:
            return 'Others'
    elif desire == 'Recreation':
        if 'Leisure & Wellness' in amenity:
            return 'Leisure & Wellness'
        elif 'Entertainment' in amenity:
            return 'Entertainment'
        elif 'Cultural' in amenity:
            return 'Cultural'
        elif 'Hotel' in amenity:
            return 'Hotel'
        else:
            return 'Others'


def get_dataset(train_file, test_file, desire, random_state=42):

    train_dataset = pd.read_csv(train_file, index_col=False)
    test_dataset = pd.read_csv(test_file, index_col=False)
    # codes, uniques = pd.factorize(train_dataset['target_amenity'])
    # amenity_mapping = {category: code for code, category in enumerate(uniques)}
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
        "age_group": {
            'Young Adult': 0, 'Adult': 1, 'Middle Age': 2, 'Senior': 3, 'Elderly': 4}
    }

    # train_dataset = train_dataset[train_dataset['target_amenity'].isin(
    #     amenity_mapping.keys())]
    # test_dataset = test_dataset[test_dataset['target_amenity'].isin(
    #     amenity_mapping.keys())]

    train_dataset['target_amenity'] = train_dataset['target_amenity'].apply(
        lambda x: map_amenity(x, desire))
    test_dataset['target_amenity'] = test_dataset['target_amenity'].apply(
        lambda x: map_amenity(x, desire))

    # drop target amenity == 'Others'
    train_dataset = train_dataset[train_dataset['target_amenity'] != 'Others']

    ids = np.random.RandomState(random_state).permutation(
        len(train_dataset) + len(test_dataset))

    train_dataset['person_id'] = ids[:len(train_dataset)]
    test_dataset['person_id'] = ids[len(train_dataset):]

    return train_dataset, test_dataset, mapping


def get_data(dataset, mapping, y_prop):
    data = dataset.copy()
    for key in mapping.keys():
        data[key] = data[key].map(mapping[key])

    X_props = ['person_id', 'age_group', 'family_structure',
               'income_group', 'household_size', 'vehicles']

    y_prop = ['person_id', y_prop]

    X = data[X_props]
    y = data[y_prop]
    return X, y
