import pandas as pd
import os
from tqdm import tqdm
from .init_settings import init_settings

person_props = ['age', 'individual_income',
                'household_size', 'family_structure', 'vehicles']

desire_prop = ['travel_purpose']

intension_prop = ['top_amenity', 'sub_amenity', 'mode',
                  'distance_miles', 'duration_minutes', 'location_name']


def load_orig_data(data_dir):
    orig_path = os.path.join(data_dir,
                             'orig/trips_th_with_pop.csv')
    trips_th_with_pop = pd.read_csv(orig_path, index_col=False)
    # shuffle the data
    trips_th_with_pop = trips_th_with_pop.sample(
        frac=1, random_state=42).reset_index(drop=True)
    # remove the rows where family structure equals to 'GQ'
    trips_th_with_pop = trips_th_with_pop[trips_th_with_pop['family_structure'] != 'GQ']
    # remove the rows where vehicles equals to 'GQ'
    trips_th_with_pop = trips_th_with_pop[trips_th_with_pop['vehicles'] != 'GQ']
    trips_th_with_pop['vehicles'] = trips_th_with_pop['vehicles'].replace(
        'zero', '0')
    # remove the rows where mode equals to 'Biking'
    trips_th_with_pop = trips_th_with_pop[trips_th_with_pop['mode'] != 'Biking']
    # remove blanks in amenity
    trips_th_with_pop['top_amenity'] = trips_th_with_pop['top_amenity'].str.strip()
    trips_th_with_pop['sub_amenity'] = trips_th_with_pop['sub_amenity'].str.strip()

    train_dataset = trips_th_with_pop[:int(
        len(trips_th_with_pop)*0.8)].reset_index()
    test_dataset = trips_th_with_pop[int(
        len(trips_th_with_pop)*0.8):].reset_index()
    return train_dataset, test_dataset


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
    else:
        return amenity


def prepare_train_data(data_dir, sample_num, desire='Eat'):

    train_dataset, _ = load_orig_data(data_dir)

    data_folder = os.path.join(data_dir, f'train/{sample_num}/{desire}')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if desire:
        train_data = train_dataset[train_dataset['travel_purpose']
                                   == desire].reset_index()
        train_data = train_data[:sample_num]
    else:
        # shuffle the data
        train_data = train_dataset.sample(
            frac=1, random_state=42).reset_index(drop=True)
        train_data = train_data[:sample_num]

    person_df = pd.DataFrame(columns=['id', 'age', 'individual_income',
                             'household_size', 'family_structure', 'vehicles', 'description'])
    desire_df = pd.DataFrame(columns=['id', 'desire', 'description'])
    intention_df = pd.DataFrame(columns=['id', 'target_amenity', 'mode',
                                'distance_miles', 'duration_minutes', 'location_name', 'description'])

    want_to_edge_df = pd.DataFrame(columns=['source', 'target', 'type'])
    go_to_edge_df = pd.DataFrame(columns=['source', 'target', 'type'])

    for idx, row in tqdm(train_data.iterrows(), desc=f'preparing train data...', total=sample_num):
        person_id = row['person_id']
        person_description = f"A {row['age']} year old person, living in a {row['family_structure']} family with {row['household_size']} members. The person has {row['vehicles']} vehicles and an annual income of {row['individual_income']} dollars."

        desire = row['travel_purpose']
        desire_description = f"This person wants to {row['travel_purpose']}"

        intention_id = row['activity_id']
        intention_description = f"This person goes to {row['location_name']} by {row['mode']}, this place is in the amenity category of {row['top_amenity']}/{row['sub_amenity']}. The distance is {row['distance_miles']} miles. It takes {row['duration_minutes']} minutes."

        person_df.loc[len(person_df)] = [f"Person_{person_id}", row['age'], row['individual_income'],
                                         row['household_size'], row['family_structure'], row['vehicles'], person_description]

        desire_df.loc[len(desire_df)] = [
            f"Desire_{person_id}_{desire}", row['travel_purpose'], desire_description]

        amenity = map_amenity(row['top_amenity'] +
                              '/'+row['sub_amenity'], desire)
        intention_df.loc[len(intention_df)] = [f"Internsion_{intention_id}", amenity, row['mode'], row
                                               ['distance_miles'], row['duration_minutes'], row['location_name'], intention_description]

        want_to_edge_df.loc[len(want_to_edge_df)] = [
            f"Person_{person_id}", f"Desire_{person_id}_{desire}", 'want_to']

        go_to_edge_df.loc[len(go_to_edge_df)] = [
            f"Desire_{person_id}_{desire}", f"Internsion_{intention_id}", 'go_to']

    person_df = person_df.drop_duplicates()
    desire_df = desire_df.drop_duplicates()
    intention_df = intention_df.drop_duplicates()
    want_to_edge_df = want_to_edge_df.drop_duplicates()
    go_to_edge_df = go_to_edge_df.drop_duplicates()

    person_df.to_csv(
        f'{data_folder}/person.csv', index=False)
    desire_df.to_csv(
        f'{data_folder}/desire.csv', index=False)
    intention_df.to_csv(
        f'{data_folder}/intention.csv', index=False)
    want_to_edge_df.to_csv(
        f'{data_folder}/want_to_edge.csv', index=False)
    go_to_edge_df.to_csv(
        f'{data_folder}/go_to_edge.csv', index=False)

    return data_folder


def prepare_test_data(data_dir, desire='Eat', test_num=1000):

    _, test_dataset = load_orig_data(data_dir)

    data_folder = os.path.join(data_dir, f'test')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    test_data = test_dataset[test_dataset['travel_purpose']
                             == desire].reset_index()
    income_bins = [-float('inf'), 0, 30000, 60000, 90000, 120000, float('inf')]
    income_labels = ['p.income<0', 'p.income>=0 AND p.income<30000', 'p.income>=30000 AND p.income<60000',
                     'p.income>=60000 AND p.income<90000', 'p.income>=90000 AND p.income<120000', 'p.income>=120000']
    age_bins = [-float('inf'), 18, 30, 40, 50, 60, float('inf')]
    age_labels = ['p.age<18', 'p.age>=18 AND p.age<30', 'p.age>=30 AND p.age<40',
                  'p.age>=40 AND p.age<50', 'p.age>=50 AND p.age<60', 'p.age>60']

    test_data['income_group'] = pd.cut(
        test_data['individual_income'], bins=income_bins, labels=income_labels)
    test_data['age_group'] = pd.cut(
        test_data['age'], bins=age_bins, labels=age_labels)
    test_data['target_amenity'] = test_data['top_amenity'] + \
        '/'+test_data['sub_amenity']
    test_data['target_amenity'] = test_data['target_amenity'].apply(
        lambda x: map_amenity(x, desire))
    test_data['distance_miles'] = test_data['distance_miles']//init_settings.distance_step * \
        init_settings.distance_step
    test_data['duration_minutes'] = test_data['duration_minutes']//init_settings.duration_step * \
        init_settings.duration_step
    # remove agen_group == 'Teen', because the training data does not have enough samples
    test_data = test_data[test_data['age_group'] != 'p.age<18']
    test_data = test_data[:test_num]

    for idx, row in tqdm(test_data.iterrows(), desc=f'preparing test data...', total=test_num):
        cypher = f'''
        MATCH (p:Person)
        WHERE ({row['age_group']}) OR ({row['income_group']})
        WITH p ORDER BY rand() LIMIT 50
        MATCH (p)-[r1:WANT_TO]-(d:Desire {{desire:'{desire}'}})-[r2:GO_TO]-(i:Intention)
        RETURN COLLECT(DISTINCT p) AS person, 
            COLLECT(DISTINCT d) AS desire, 
            COLLECT(DISTINCT i) AS intention,
            COLLECT(DISTINCT r1) AS want_to,
            COLLECT(DISTINCT r2) AS go_to
        '''
        test_data.loc[idx, 'cypher'] = cypher

    test_data = test_data[['person_id', 'travel_purpose', 'target_amenity', 'mode',
                           'distance_miles', 'duration_minutes', 'age', 'individual_income', 'household_size', 'family_structure', 'vehicles',
                           'cypher']]

    test_data_path = f"{data_folder}/{desire}.csv"
    if not os.path.exists(test_data_path):
        test_data.to_csv(test_data_path, index=False)

    return test_data_path
