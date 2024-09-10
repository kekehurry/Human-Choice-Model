import numpy as np
import pandas as pd


def get_dataset(train_file, test_file, random_state=42):

    train_dataset = pd.read_csv(train_file, index_col=False)
    test_dataset = pd.read_csv(test_file, index_col=False)
    # codes, uniques = pd.factorize(train_dataset['target_amenity'])
    # amenity_mapping = {category: code for code, category in enumerate(uniques)}
    amenity_mapping = {
        'F&B Eatery/Limited-Service Restaurants': 0,
        'F&B Eatery/Snack and Nonalcoholic Beverage Bars': 1,
        'F&B Eatery/Full-Service Restaurants': 2,
        'F&B Eatery/Drinking Places': 3,
        # 'F&B Eatery/Special Food Services': 4,
    }
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
            'Teen': 0, 'Young Adult': 1, 'Adult': 2, 'Middle Age': 3, 'Senior': 4, 'Elderly': 5}
    }

    train_dataset = train_dataset[train_dataset['target_amenity'].isin(
        amenity_mapping.keys())]
    test_dataset = test_dataset[test_dataset['target_amenity'].isin(
        amenity_mapping.keys())]

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
