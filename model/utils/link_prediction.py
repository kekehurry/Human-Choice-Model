import numpy as np
from collections import defaultdict
import networkx as nx
from .init_settings import init_settings


def get_edge_counter(query_results):
    want_to = query_results['want_to']
    go_to = query_results['go_to']
    counter = defaultdict(int)
    counter_sum = defaultdict(int)
    for edge_w in want_to:
        person = edge_w[0]["id"]
        desire_1 = edge_w[2]["id"]
        for edge_g in go_to:
            desire_2 = edge_g[0]["id"]
            intention = edge_g[2]
            if desire_1 == desire_2:
                duration = intention["duration_minutes"]//init_settings.duration_step * \
                    init_settings.duration_step + 1
                distance = intention["distance_miles"]//init_settings.distance_step * \
                    init_settings.distance_step + 1
                mode = intention["mode"]
                amenity = intention["target_amenity"]
                counter[(person, f"Duration_{duration}_min", 'durantion')] += 1
                counter[(person, mode, 'mode')] += 1
                counter[(person, amenity, 'amenity')] += 1
                counter[(
                    person, f"Distance_{distance:.1f}_miles", 'distance')] += 1
                counter_sum[(person, 'durantion')] += 1
                counter_sum[(person, 'mode')] += 1
                counter_sum[(person, 'amenity')] += 1
                counter_sum[(person, 'distance')] += 1
    return counter, counter_sum


def create_behavior_subgraph(query_results):
    # Get behavior subgraph
    person = query_results['person']
    desire = query_results['desire']
    intention = query_results['intention']
    G = nx.Graph()
    # Add agent node
    agent_id = np.random.randint(1, 10000)
    G.add_node(f'Agent_{agent_id:5d}', color="#76FF03",
               label="Agent", type="Agent")
    # Add nodes
    for node in person:
        G.add_node(node["id"], **node, color="purple",
                   label="Person", type="Person")
        G.add_edge(f'Agent_{agent_id:5d}', node["id"], label="SIMILAR_TO",
                   type="SIMILAR_TO", weight=node["similarity_score"])
    for node in intention:
        duration = node["duration_minutes"]//init_settings.duration_step * \
            init_settings.duration_step + 1
        distance = node["distance_miles"]//init_settings.distance_step * \
            init_settings.distance_step + 1
        mode = node["mode"]
        amenity = node["target_amenity"]
        G.add_node(f"Duration_{duration}_min", color="blue",
                   label=f"Duration_{duration}_min", type="Duration")
        G.add_node(mode, color="green", label=node["mode"], type="Mode")
        G.add_node(amenity, color="red",
                   label=node["target_amenity"], type="Amenity")
        G.add_node(f"Distance_{distance:.1f}_miles", color="orange",
                   label=f"Distance_{distance:.1f}_miles", type="Distance")
    # Add edges
    counter, counter_sum = get_edge_counter(query_results)
    for (peson, choice, type), v in counter.items():
        sum_v = counter_sum[(peson, type)]
        G.add_edge(peson, choice, label="CHOOSE",
                   type="CHOOSE", weight=v/sum_v)
    return G


def cal_weighted_jaccard_coeff(G, agent, choice):
    common_neighbors = list(nx.common_neighbors(G, agent, choice))
    total_neighbors = len(list(nx.neighbors(G, agent)))
    weight_sum = 0
    for n in common_neighbors:
        edge_ap = G.edges[agent, n]['weight']
        edge_pc = G.edges[n, choice]['weight']
        weight_sum += (edge_ap+edge_pc)/2
    return weight_sum/total_neighbors


def get_recommendation(G, choice_type):
    agents = [n for n, d in G.nodes(data=True) if d['type'] == 'Agent']
    choices = [n for n, d in G.nodes(data=True) if d['type'] == choice_type]
    recommendation = []
    for choice in choices:
        score = cal_weighted_jaccard_coeff(G, agents[0], choice)
        recommendation.append({
            "choice": choice,
            "score": round(score, 2)
        })
    recommendation = sorted(
        recommendation, key=lambda item: item['score'], reverse=True)
    return recommendation
