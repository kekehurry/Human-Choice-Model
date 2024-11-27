from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import OllamaEmbeddings
import uuid
from .init_settings import init_settings


def get_similarity_score(profile, query_results):
    people = query_results['person']
    description_embedding_pairs = []
    for p in people:
        description_embedding_pairs.append((p['id'], p['embedding']))

    index_name = "sub_person_index"
    result = init_settings.graph.query(
        f"SHOW INDEXES WHERE name = '{index_name}'")
    if result:
        sub_person_index = Neo4jVector.from_existing_graph(
            url=init_settings.url,
            username=init_settings.username,
            password=init_settings.password,
            embedding=init_settings.embedding_model,
            index_name=index_name,
        )
    else:
        sub_person_index = Neo4jVector.from_embeddings(
            url=init_settings.url,
            username=init_settings.username,
            password=init_settings.password,
            text_embeddings=description_embedding_pairs,
            embedding=init_settings.embedding_model,
            pre_delete_collection=True,
            index_name=index_name
        )
    results = sub_person_index.similarity_search_with_score(
        query=profile, k=len(people))
    results = [{"id": r.page_content, "similarity_score": score}
               for r, score in results]
    for p in people:
        for r in results:
            if p['id'] == r['id']:
                p['similarity_score'] = r['similarity_score']
    query_results['person'] = people
    return query_results


def get_similar_nodes(profile, query_results, k=None):
    people = query_results['person']
    description_embedding_pairs = []
    if k is None:
        k = len(people)
    for p in people:
        description_embedding_pairs.append((p['id'], p['embedding']))
    index_name = "sub_person_index"
    result = init_settings.graph.query(
        f"SHOW INDEXES WHERE name = '{index_name}'")
    if result:
        sub_person_index = Neo4jVector.from_existing_graph(
            url=init_settings.url,
            username=init_settings.username,
            password=init_settings.password,
            embedding=init_settings.embedding_model,
            index_name=index_name,
        )
    else:
        sub_person_index = Neo4jVector.from_embeddings(
            url=init_settings.url,
            username=init_settings.username,
            password=init_settings.password,
            text_embeddings=description_embedding_pairs,
            embedding=init_settings.embedding_model,
            pre_delete_collection=True,
            index_name=index_name
        )
    results = sub_person_index.similarity_search_with_score(
        query=profile, k=k)
    results = [{"id": r.page_content, "similarity_score": score}
               for r, score in results]
    return results
