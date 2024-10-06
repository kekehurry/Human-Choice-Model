
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
import uuid
from collections import Counter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import random

# neo4j configuration
url = "bolt://localhost:7687"
username = "neo4j"
password = "neo4jgraph"

graph = Neo4jGraph(
    url=url,
    username=username,
    password=password,
    enhanced_schema=True,
)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
choice_model = ChatOllama(model='llama3.1', format="json")
app = Flask(__name__)
CORS(app)


def get_similar_node_ids(profile, desire, k=50):
    query = f'''
MATCH (p)-[r1:WANT_TO]->(d:Desire {{desire: '{desire}'}})-[r2:GO_TO]->(i:Intention)
WITH 
    COLLECT(DISTINCT p {{.*, type: 'Person', id: p.id}}) AS person
return person
'''
    query_results = graph.query(query)
    query_results = query_results[0]
    people = query_results['person']
    description_embedding_pairs = []
    for p in people:
        description_embedding_pairs.append((p['id'], p['embedding']))

    sub_person_index = Neo4jVector.from_embeddings(
        url=url,
        username=username,
        password=password,
        text_embeddings=description_embedding_pairs,
        embedding=embedding_model,
        pre_delete_collection=True,
    )
    results = sub_person_index.similarity_search_with_score(
        query=profile, k=k)
    ids = [r.page_content for r, _ in results]
    return ids


def simple_analysis(query_result):
    intentions = [n for n in query_result['nodes'] if n['type'] == 'Intention']
    people = [n for n in query_result['nodes'] if n['type'] == 'Person']
    ages = [{"name": p['id'], "value": p['age']} for p in people]
    incomes = [{"name": p['id'], "value": int(
        p['income']/1000)} for p in people]
    amenity_count = Counter([i['target_amenity'] for i in intentions])
    sum_amenity = sum(amenity_count.values())
    mobility_count = Counter([i['mode'] for i in intentions])
    sum_mobility = sum(mobility_count.values())
    amenity_choices = [{"name": k, "value": v/sum_amenity}
                       for k, v in amenity_count.items()]
    mobility_choices = [{"name": k, "value": v/sum_mobility}
                        for k, v in mobility_count.items()]
    return amenity_choices, mobility_choices, ages, incomes


def get_llm_choice(profile, desire, choice_type, recommendation, city='Boston'):
    choice_template = """
What {choice_type} would a person who is {profile} and lives in {city} choose, when he want to {desire}?
-
Context: This is the reference of people who have similar profile with the given profile in Boston and their choice.
if scored is provided, higher socre of related choice in context means more likely to be chosen by people with similar profile.
{recommendation}

Note: 
- Output the weights of all options provided in a json format.
- You should consider the profile, the choices other people with similar profile did, and the possible cultural difference between Boston and {city}.
- The sum of all weights should equals to 1.

Answer Format:
{{
'thought': 'Based on the profile, xxxx. Consider the choices of people with similar profile, xxx. And the culture in {city}, xxx. I think xxx',
'final_answer':[
{{'choice': option1. 'weight': possibility for option1}},{{'choice': option1. 'weight': possibility for option2}},...
],
}}
"""
    choose_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert analyst of human behavior, given the context of profile and behaviors of similar people, evaluate the possibility of different options.",
            ),
            ("human", choice_template),
        ]
    )
    choice_response = (
        choose_prompt
        | choice_model.bind()
        | StrOutputParser()
    )
    answer = choice_response.invoke({
        "profile": profile,
        "desire": desire,
        "choice_type": choice_type,
        "recommendation": recommendation,
        "city": city
    })
    try:
        llm_choice = json.loads(answer)
        options = [item['choice'].strip()
                   for item in llm_choice['final_answer']]
        weights = [item['weight']
                   for item in llm_choice['final_answer']]
        choice = random.choices(options, weights=weights, k=1)[0]
    except Exception as e:
        return None
    return choice


@app.route('/init', methods=['GET'])
def init():
    cypher_query = '''
MATCH (p:Person)
WITH p ORDER BY rand() LIMIT 1000

// Match the original relationships for the randomly selected persons
MATCH (p)-[r1:WANT_TO]->(d:Desire)-[r2:GO_TO]->(i:Intention)
OPTIONAL MATCH (p)-[r3:SIMILAR_TO]->(s:Person)

// Also match the relationships for the similar persons
OPTIONAL MATCH (s)-[r4:WANT_TO]->(sd:Desire)-[r5:GO_TO]->(si:Intention)

RETURN 
    // Collect original nodes
    COLLECT(DISTINCT p {.*, type: 'Person', id: p.id, name: p.description, embedding: null}) + 
    COLLECT(DISTINCT d {.*, type: 'Desire', id: d.id, name: d.desire, embedding: null}) + 
    COLLECT(DISTINCT i {.*, type: 'Intention', id: i.id, name: i.description, embedding: null}) +
    // Collect similar nodes
    COLLECT(DISTINCT s {.*, type: 'Person', id: d.id, name: s.description, embedding: null}) +
    COLLECT(DISTINCT sd {.*, type: 'Desire', id: sd.id, name: sd.desire, embedding: null}) +
    COLLECT(DISTINCT si {.*, type: 'Intention', id: si.id, name: si.description, embedding: null}) AS nodes,

    // Collect original links
    COLLECT(DISTINCT { source: p.id, target: d.id, type: 'WANT_TO' }) +
    COLLECT(DISTINCT { source: d.id, target: i.id, type: 'GO_TO' }) + 
    COLLECT(DISTINCT { source: p.id, target: s.id, type: 'SIMILAR_TO' }) +
    // Collect similar person links
    COLLECT(DISTINCT { source: s.id, target: sd.id, type: 'WANT_TO' }) +
    COLLECT(DISTINCT { source: sd.id, target: si.id, type: 'GO_TO' }) AS links
'''
    query_result = graph.query(cypher_query)
    query_result = query_result[0]
    amenity_choices, mobility_choices, ages, incomes = simple_analysis(
        query_result)
    return jsonify({
        'graph': query_result,
        'amenity_choices': amenity_choices,
        'mobility_choices': mobility_choices,
        'ages': ages,
        'incomes': incomes,
    })


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    profile = data['profile']
    desire = data['desire']
    k = data['k']
    ids = get_similar_node_ids(profile, desire, k)
    cypher_query = f'''
MATCH (p:Person)
WHERE p.id IN {ids}
MATCH (p)-[r1:WANT_TO]->(d:Desire {{desire: '{desire}'}})-[r2:GO_TO]->(i:Intention)
MATCH (p)-[r3:SIMILAR_TO]->(s:Person)
MATCH (s)-[r4:WANT_TO]->(sd:Desire {{desire: '{desire}'}})-[r5:GO_TO]->(si:Intention)

WITH 
    // Collect original links
    COLLECT(DISTINCT {{ source: p.id, target: d.id, type: 'WANT_TO' }}) + 
    COLLECT(DISTINCT {{ source: d.id, target: i.id, type: 'GO_TO' }}) +
    COLLECT(DISTINCT {{ source: p.id, target: s.id, type: 'SIMILAR_TO' }}) +
    // Collect similar person links
    COLLECT(DISTINCT {{ source: s.id, target: sd.id, type: 'WANT_TO' }}) +
    COLLECT(DISTINCT {{ source: sd.id, target: si.id, type: 'GO_TO' }}) 
    AS links,

    // Collect original and similar nodes
    COLLECT(DISTINCT p {{.*, type: 'Person', id: p.id, name: p.description, embedding: null}}) + 
    COLLECT(DISTINCT d {{.*, type: 'Desire', id: d.id, name: d.desire, embedding: null}}) + 
    COLLECT(DISTINCT i {{.*, type: 'Intention', id: i.id, name: i.description, embedding: null}}) + 
    COLLECT(DISTINCT s {{.*, type: 'Person', id: s.id, name: s.description, embedding: null}}) +
    COLLECT(DISTINCT sd {{.*, type: 'Desire', id: sd.id, name: sd.desire, embedding: null}}) +
    COLLECT(DISTINCT si {{.*, type: 'Intention', id: si.id, name: si.description, embedding: null}}) 
    AS nodes

WITH 
    nodes, links,

    // Extract IDs of linked nodes
    [l IN links | l.source] + [l IN links | l.target] AS linked_node_ids

// Filter out isolated nodes
WITH [n IN nodes WHERE n.id IN linked_node_ids] AS filtered_nodes, links

RETURN filtered_nodes AS nodes, links
'''
    query_result = graph.query(cypher_query)
    query_result = query_result[0]
    amenity_choices, mobility_choices, ages, incomes = simple_analysis(
        query_result)
    amenity_final_choice = get_llm_choice(
        profile, desire, 'Amenity', amenity_choices)
    mobility_final_choice = get_llm_choice(
        profile, desire, 'Mobility', mobility_choices)
    # Add agent node
    agent_id = 'Agent_' + str(uuid.uuid4())
    node_ids = [n['id'] for n in query_result['nodes']]
    query_result['nodes'].append({
        'id': agent_id,
        'name': profile,
        'type': 'Agent',
    })
    for id in ids:
        if id in node_ids:
            query_result['links'].append({
                'source': agent_id,
                'target': id,
                'type': 'SIMILAR_TO'
            })
    return jsonify({
        'graph': query_result,
        'amenity_choices': amenity_choices,
        'mobility_choices': mobility_choices,
        "amenity_final_choice": amenity_final_choice,
        "mobility_final_choice": mobility_final_choice,
        'ages': ages,
        'incomes': incomes,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
