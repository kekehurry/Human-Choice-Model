from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from neo4j import GraphDatabase


class Settings:

    url = "bolt://localhost:7687"
    username = "neo4j"
    password = "neo4jgraph"

    driver = GraphDatabase.driver(url, auth=(username, password))
    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password,
        enhanced_schema=True,
    )

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    cypher_model = ChatOllama(model='llama3.1', temperature=0)
    choice_model = ChatOllama(model='llama3.1', format="json")

    duration_step = 5
    distance_step = 0.5


init_settings = Settings()
