from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from .init_settings import init_settings
from langchain.vectorstores import FAISS
import faiss
import numpy as np


import numpy as np
import faiss


def get_similarity_score(profile, query_results):
    people = query_results['person']
    embeddings = []
    ids = []

    for p in people:
        embeddings.append(p['embedding'])
        ids.append(p['id'])

    # Normalize embeddings for cosine similarity
    embeddings = np.array(embeddings).astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Create FAISS index for cosine similarity
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)

    # Normalize profile embedding
    profile_embedding = init_settings.embedding_model.embed_query(profile)
    profile_norm = np.linalg.norm(profile_embedding)
    normalized_profile_embedding = profile_embedding / profile_norm

    # Perform similarity search
    D, I = index.search(np.array([normalized_profile_embedding]), len(people))

    # Collect results
    results = [{"id": ids[i], "similarity_score": D[0][j]}
               for j, i in enumerate(I[0])]

    for p in people:
        for r in results:
            if p['id'] == r['id']:
                p['similarity_score'] = r['similarity_score']
    query_results['person'] = people
    return query_results


def get_similar_nodes(profile, query_results, k=None):
    people = query_results['person']
    embeddings = []
    ids = []
    if k is None:
        k = len(people)
    for p in people:
        embeddings.append(p['embedding'])
        ids.append(p['id'])

    # Normalize embeddings for cosine similarity
    embeddings = np.array(embeddings).astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Create FAISS index for cosine similarity
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)

    # Normalize profile embedding
    profile_embedding = init_settings.embedding_model.embed_query(profile)
    profile_norm = np.linalg.norm(profile_embedding)
    normalized_profile_embedding = profile_embedding / profile_norm

    # Perform similarity search
    D, I = index.search(np.array([normalized_profile_embedding]), len(people))

    # Collect results
    results = [{"id": ids[i], "similarity_score": D[0][j]}
               for j, i in enumerate(I[0])]

    return results


# def get_similarity_score(profile, query_results):
#     people = query_results['person']
#     description_embedding_pairs = []
#     for p in people:
#         description_embedding_pairs.append((p['id'], p['embedding']))

#     sub_person_index = Neo4jVector.from_embeddings(
#         url=init_settings.url,
#         username=init_settings.username,
#         password=init_settings.password,
#         text_embeddings=description_embedding_pairs,
#         embedding=init_settings.embedding_model,
#         pre_delete_collection=True,
#     )
#     results = sub_person_index.similarity_search_with_score(
#         query=profile, k=len(people))
#     results = [{"id": r.page_content, "similarity_score": score}
#                for r, score in results]

#     for p in people:
#         for r in results:
#             if p['id'] == r['id']:
#                 p['similarity_score'] = r['similarity_score']
#     query_results['person'] = people
#     return query_results


# def get_similar_nodes(profile, query_results, k=None):
#     people = query_results['person']
#     description_embedding_pairs = []
#     if k is None:
#         k = len(people)
#     for p in people:
#         description_embedding_pairs.append((p['id'], p['embedding']))
#     sub_person_index = Neo4jVector.from_embeddings(
#         url=init_settings.url,
#         username=init_settings.username,
#         password=init_settings.password,
#         text_embeddings=description_embedding_pairs,
#         embedding=init_settings.embedding_model,
#         pre_delete_collection=True,
#     )
#     results = sub_person_index.similarity_search_with_score(
#         query=profile, k=k)
#     results = [{"id": r.page_content, "similarity_score": score}
#                for r, score in results]
#     return results
