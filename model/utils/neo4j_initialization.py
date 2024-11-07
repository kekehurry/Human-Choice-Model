from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import OllamaEmbeddings
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from .init_settings import init_settings


def clear_database():
    def clear(tx):
        # Delete all relationships
        tx.run("MATCH (()-[r]->() ) DELETE r")
        # Delete all nodes
        tx.run("MATCH (n) DELETE n")
    with init_settings.driver.session() as session:
        session.execute_write(clear)


def init_database(train_dir):
    person_df = pd.read_csv(f'{train_dir}/person.csv')
    desire_df = pd.read_csv(f'{train_dir}/desire.csv')
    itention_df = pd.read_csv(f'{train_dir}/intention.csv')
    go_to_edge_df = pd.read_csv(f'{train_dir}/go_to_edge.csv')
    want_to_edge_df = pd.read_csv(f'{train_dir}/want_to_edge.csv')

    def create_person(tx, id, age, income, household_size, family_structure, vehicles, name, description):
        tx.run("CREATE (a:Person {id: $id, age: $age, income: $income, household_size: $household_size, family_structure: $family_structure, vehicles: $vehicles, name: $name, description: $description})",
               id=id, age=age, income=income, household_size=household_size, family_structure=family_structure, vehicles=vehicles, name=name, description=description)

    def create_desire(tx, id, desire, description):
        tx.run("CREATE (a:Desire {id: $id, desire: $desire, description: $description})",
               id=id, desire=desire, description=description)

    def create_intention(tx, id, target_amenity, mode, distance_miles, duration_minutes, location_name, description):
        tx.run("CREATE (a:Intention {id: $id, target_amenity: $target_amenity, mode: $mode, distance_miles: $distance_miles, duration_minutes: $duration_minutes, location_name: $location_name, description: $description})",
               id=id, target_amenity=target_amenity, mode=mode, distance_miles=distance_miles, duration_minutes=duration_minutes, location_name=location_name, description=description)

    def create_want_to_edge(tx, person_id, desire_id):
        tx.run("MATCH (a:Person),(b:Desire) WHERE a.id = $person_id AND b.id = $desire_id CREATE (a)-[r:WANT_TO]->(b)",
               person_id=person_id, desire_id=desire_id)

    def create_go_to_edge(tx, desire_id, intention_id):
        tx.run("MATCH (a:Desire),(b:Intention) WHERE a.id = $desire_id AND b.id = $intention_id CREATE (a)-[r:GO_TO]->(b)",
               desire_id=desire_id, intention_id=intention_id)

    with init_settings.driver.session() as session:

        for i, row in tqdm(person_df.iterrows(), desc='adding person nodes...', total=len(person_df)):
            session.execute_write(create_person, row['id'], row['age'], row['individual_income'],
                                  row['household_size'], row['family_structure'], row['vehicles'], 'Person', row['description'])

        for i, row in tqdm(desire_df.iterrows(), desc='adding desire nodes...', total=len(desire_df)):
            session.execute_write(
                create_desire, row['id'], row['desire'], row['description'])

        for i, row in tqdm(itention_df.iterrows(), desc='adding itention nodes...', total=len(itention_df)):
            session.execute_write(create_intention, row['id'], row['target_amenity'], row['mode'],
                                  row['distance_miles'], row['duration_minutes'], row['location_name'], row['description'])

        for i, row in tqdm(want_to_edge_df.iterrows(), desc='adding wan_to edges...', total=len(want_to_edge_df)):
            session.execute_write(create_want_to_edge,
                                  row['source'], row['target'])

        for i, row in tqdm(go_to_edge_df.iterrows(), desc='adding go_to edges...', total=len(go_to_edge_df)):
            session.execute_write(
                create_go_to_edge, row['source'], row['target'])


def create_index():
    retrieval_query = """
    RETURN node.description AS text, score, node {.age,.individual_income,.household_size,.family_structure,.vehicles} AS metadata
    """

    person_index = Neo4jVector.from_existing_graph(
        init_settings.embedding_model,
        url=init_settings.url,
        username=init_settings.username,
        password=init_settings.password,
        index_name='person',
        node_label="Person",
        text_node_properties=['description'],
        embedding_node_property='embedding',
        retrieval_query=retrieval_query,
    )


def prepare_neo4j(train_dir):
    clear_database()
    init_database(train_dir)
    print('creating neo4j index...')
    create_index()
    print('done!')
