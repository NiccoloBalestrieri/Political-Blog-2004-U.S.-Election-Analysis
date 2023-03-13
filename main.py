import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import infomap
import random
from igraph import Graph
import csv

import networkit as nt
import matplotlib.colors as colors
import matplotlib.cm as cm

from collections import Counter

'''
# Leggi il file CSV contenente la lista degli archi
with open('C:/Users/nicco/OneDrive/Documenti/ProgettoCSR/dataset/edge_list.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    header = next(reader)  # leggi l'intestazione del file
    edges = [row for row in reader]

# Leggi il file CSV contenente l'orientamento dei nodi
with open('C:/Users/nicco/OneDrive/Documenti/ProgettoCSR/dataset/orientation_index.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # leggi l'intestazione del file
    node_orientation = {row[0]: row[1] for row in reader}  # crea un dizionario con l'orientamento dei nodi

# Crea una nuova colonna "type" nella lista degli archi in base all'orientamento dei nodi
new_header = header + ['type']
for i, row in enumerate(edges):
    source = row[0]
    target = row[1]
    source_orientation = node_orientation.get(source, 'unknown')
    target_orientation = node_orientation.get(target, 'unknown')
    if source_orientation == 'left-leaning' and target_orientation == 'left-leaning':
        edges[i].append('red') #1 posizione
    elif source_orientation == 'right-leaning' and target_orientation == 'right-leaning':
        edges[i].append('purple') #2 posizione
    elif source_orientation == 'left-leaning' and target_orientation == 'right-leaning':
        edges[i].append('yellow') #4 posizione
    elif source_orientation == 'right-leaning' and target_orientation == 'left-leaning':
        edges[i].append('green') #3 posizione
    else:
        edges[i].append('unknown')

# Scrivi la lista degli archi con la nuova colonna "type" in un nuovo file CSV
with open('C:/Users/nicco/OneDrive/Documenti/ProgettoCSR/dataset/edge_list_with_type.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(new_header)
    writer.writerows(edges)
'''

def components(graph):
    scc = max(nx.strongly_connected_components(graph), key=len)
    wcc = max(nx.weakly_connected_components(graph), key=len)
    n_scc = nx.number_strongly_connected_components(graph)
    n_wcc = nx.number_weakly_connected_components(graph)
    return scc, wcc, n_scc, n_wcc

def write_largest_connected_component(graph, filename):
    # Calcola la più grande componente connessa
    scc = max(nx.strongly_connected_components(graph), key=len)
    print(len(scc))
    graph = graph.subgraph(scc)
    # Crea un elenco di tuple che rappresentano gli archi della rete
    edges = list(graph.edges())
    print(len(edges))
    # Salva le coppie di nodi e i relativi archi in un file CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Source', 'Target'])
        for edge in edges:
            writer.writerow(edge)

    # Apri il file CSV di input in modalità di lettura
    with open(filename, 'r') as csv_file:
        # Leggi le coppie di origine dal file CSV
        original_pairs = [tuple(row) for row in csv.reader(csv_file, delimiter=';')]

    # Modifica le coppie di origine come richiesto
    modified_pairs = [(str(pair[0]) + "A", str(pair[1]) + "A") for pair in original_pairs]

    # Apri il file CSV di output in modalità di scrittura
    with open('C:/Users/nicco/OneDrive/Documenti/GitHub/Political-Blog-2004-U.S.-Election-Analysis/dataset/largest_componentA.csv.csv', 'w', newline='') as csv_file:
        # Scrivi le coppie modificate nel file CSV di output
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerows(modified_pairs)

def degreeCentrality(graph):
    degree = nx.degree_centrality(graph)
    sorted_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_degree[:5]
    return top_5

def outDegreeCentrality(graph):
    out_degree = nx.out_degree_centrality(graph)
    # ordinamento dei nodi per centralità
    sorted_centrality = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
    # selezione dei primi 5 nodi con punteggio migliore
    top_5 = sorted_centrality[:5]
    # creazione del dizionario con i valori di node centrality
    top_5_dict = {node: score for node, score in top_5}
    # creazione del grafico a barre
    fig, ax = plt.subplots()
    plt.title("Out-Node centrality")
    positions = [1, 2, 3, 4, 5]  # posizioni sull'asse x
    clr = ['SeaGreen', 'pink', 'pink', 'pink', 'pink']
    ax.bar(positions, top_5_dict.values(), width=0.5, color = clr)
    # personalizzazione dell'asse x
    ax.set_xticks(positions)
    ax.set_xticklabels(top_5_dict.keys())
    ax.tick_params(axis='x', which='both', length=0)

    plt.show()

def inDegreeCentrality(graph):
    in_degree = nx.in_degree_centrality(graph)
    # ordinamento dei nodi per centralità
    sorted_centrality = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
    # selezione dei primi 5 nodi con punteggio migliore
    top_5 = sorted_centrality[:5]
    # creazione del dizionario con i valori di node centrality
    top_5_dict = {node: score for node, score in top_5}
    # creazione del grafico a barre
    fig, ax = plt.subplots()
    plt.title("In-Node centrality")
    positions = [1, 2, 3, 4, 5]  # posizioni sull'asse x
    clr = ['SeaGreen', 'MediumSeaGreen', 'MediumSeaGreen', 'MediumSeaGreen', 'PaleGreen']
    ax.bar(positions, top_5_dict.values(), width=0.5, color = clr)
    # personalizzazione dell'asse x
    ax.set_xticks(positions)
    ax.set_xticklabels(top_5_dict.keys())
    ax.tick_params(axis='x', which='both', length=0)

    plt.show()

def betweennessCentrality(graph):
    betweenness = nx.betweenness_centrality(graph)
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_betweenness[:5]
    # creazione del dizionario con i valori di node centrality
    top_5_dict = {node: score for node, score in top_5}
    # creazione del grafico a barre
    fig, ax = plt.subplots()
    plt.title("Betweenness centrality")
    positions = [1, 2, 3, 4, 5]  # posizioni sull'asse x
    clr = ['purple', 'fuchsia', 'salmon', 'pink', 'pink']
    ax.bar(positions, top_5_dict.values(), width=0.5, color = clr)
    # personalizzazione dell'asse x
    ax.set_xticks(positions)
    ax.set_xticklabels(top_5_dict.keys())
    ax.tick_params(axis='x', which='both', length=0)
    
    plt.show()

def closenessCentrality(graph):
    closeness = nx.closeness_centrality(graph)
    sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_closeness[:5]
    top_5_dict = {node: score for node, score in top_5}
    # creazione del grafico a barre
    fig, ax = plt.subplots()
    plt.title("Closeness centrality")
    positions = [1, 2, 3, 4, 5]  # posizioni sull'asse x
    ax.bar(positions, top_5_dict.values(), width=0.5, color = 'green')
    # personalizzazione dell'asse x
    ax.set_xticks(positions)
    ax.set_xticklabels(top_5_dict.keys())
    ax.tick_params(axis='x', which='both', length=0)
    
    plt.show()

def defineCommunities(graph):
    communities = np.zeros(len(graph.nodes()))
    for i in range(0, len(graph.nodes()), 5):
        communities[i:i+5] = i//5
    return communities

def modularity(graph, communities):
    A = nx.adjacency_matrix(graph).todense()
    nodes = len(graph.nodes())
    edges = len(graph.edges())
    Q = 0
    for i in range(nodes):
        ki = sum(A[i])
        for j in range(nodes):
            kj = sum(A[j])
            if communities[i] == communities[j]:
                Q += A[i,j] - ki*kj/(2*edges)
    return Q/(2*edges)

def plotInDegreeDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.in_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    plt.title("In-Degree distribution")
    plt.xlabel("k")
    plt.ylabel(r'$P_{in}$(k)')
    plt.plot(deg, cnt, '-', markersize=8)
    plt.show()

def plotOutDegreeDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.out_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    plt.title("Out-Degree distribution")
    plt.xlabel("k")
    plt.ylabel(r'$P_{out}$(k)')
    plt.plot(deg, cnt, '-', markersize=8)
    plt.show()

def plotInDegreeCumulativeDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.in_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    cdf = np.cumsum(cnt)
    
    #plt.figure(figsize=(12, 8))
    plt.title("In-Degree cumulated distribution")
    plt.xlabel("k")
    plt.ylabel(r'$\bar{P}_{in}$(k)')
    plt.plot(deg, cdf, '-', markersize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def plotOutDegreeCumulativeDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.out_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    cdf = np.cumsum(cnt)
    
    #plt.figure(figsize=(12, 8))
    plt.title("Out-Degree cumulated distribution")
    plt.xlabel("k")
    plt.ylabel(r'$\bar{P}_{out}$(k)')
    plt.plot(deg, cdf, '-', markersize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def pageRank(graph):
    # Compute the PageRank score for each node
    pr = nx.pagerank(graph, alpha=0.85, tol=0.001)
    # Get the top 5 nodes with the highest PageRank score
    top_5 = sorted(pr, key=pr.get, reverse=True)[:5]
    top_5_dict = {}
    for node in top_5:
        top_5_dict[node] = pr[node]
    # creazione del grafico a barre
    fig, ax = plt.subplots()
    plt.title("PageRank centrality")
    positions = [1, 2, 3, 4, 5]  # posizioni sull'asse x
    clr = ['purple', 'fuchsia', 'salmon', 'pink', 'pink']
    ax.bar(positions, top_5_dict.values(), width=0.5, color = clr)
    # personalizzazione dell'asse x
    ax.set_xticks(positions)
    ax.set_xticklabels(top_5_dict.keys())
    ax.tick_params(axis='x', which='both', length=0)

    plt.show()

def inverse_community_mapping(partition):
    partition_mapping = {}
    internal_degrees = {}
    for c in range(min(partition.values()),max(partition.values()) + 1):
        partition_mapping[c] = [k for k, v in partition.items() if v == c]
        internal_degrees[c] = 0
    return partition_mapping, internal_degrees

def compute_k_core_decomposition(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph)) 
    core_numbers = nx.algorithms.core.core_number(graph)
    print(len(inverse_community_mapping(core_numbers)[0][2]))
    kcoredf = pd.DataFrame({"Id":core_numbers.keys(), "kShellId":core_numbers.values()})
    kcoredf.to_csv("C:/Users/nicco/OneDrive/Documenti/GitHub/Political-Blog-2004-U.S.-Election-Analysis/dataset/Politican_Blogs_kcoreDecomposition.csv", index=False)

def kcore(graph):
    # remove self-loops
    graph.remove_edges_from(nx.selfloop_edges(graph)) 
    # Calcola il grado di core per ogni nodo
    coreness = nx.core_number(graph)
    # Trova il massimo grado di core tra tutti i nodi
    max_coreness = max(coreness.values())
    # Il numero di shells corrisponde al massimo grado di core trovato
    num_shells = max_coreness + 1

    return num_shells

def drawCoreDecomposition(graph):
    # Load the data from the csv file
    df = pd.read_csv("C:/Users/nicco/OneDrive/Documenti/GitHub/Political-Blog-2004-U.S.-Election-Analysis/dataset/Politican_Blogs_kcoreDecomposition.csv")
    # Define the number of shells (43 in your case)
    num_shells = kcore(graph)
    # Define the radius of the smallest and largest circles
    min_radius = 0.1
    max_radius = 0.5
    # Create the figure and the axis
    fig, ax = plt.subplots(figsize = (8, 8))
    cmap = plt.cm.get_cmap('coolwarm')
    norm = plt.Normalize(df['kShellId'].min(), df['kShellId'].max())
    # Create the circles for each k-shell id
    for i in range(num_shells):
        radius = min_radius + i * (max_radius - min_radius) / num_shells
        circle = plt.Circle((0, 0), radius, fill=False)
        ax.add_artist(circle)
    # Draw the nodes on the corresponding circle
    for i, row in df.iterrows():
        x = 0
        y = 0
        k_shell_id = row["kShellId"]
        radius = max_radius - k_shell_id * (max_radius - min_radius) / num_shells  # invert the order of calculation
        theta = i * (2 * 3.14159) / len(df)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        # Map the k-shell id to a color using the colormap
        color = cmap(k_shell_id / num_shells)
        ax.scatter(x, y, color=color, s=8)

    # Set the axis limits and turn off the axis labels
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.axis("off")
    # Add the color legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal')
    cbar.set_label('K-shell ID')
    # Adjust the spacing between the subplots to avoid node overlapping
    plt.subplots_adjust(wspace=20.0, hspace=20.0)
    # Show the plot
    plt.show()

def linkPrediction(graph):
    # crea un grafo diretto in igraph
    g = Graph.TupleList(graph.itertuples(index=False), directed=True)
    # calcola il punteggio di preferential attachment per tutte le coppie di vertici
    pa_scores = [(f"{i}-{j}", g.degree(i) * g.degree(j)) for i in range(g.vcount()) for j in range(g.vcount()) if i != j]
    # seleziona solo i 5 punteggi migliori
    top_5_pa = sorted(pa_scores, key=lambda x: x[1], reverse=True)[:5]
    pa_df = pd.DataFrame(top_5_pa, columns=['Nodes', 'Preferential Attachment Index'])
    # creazione dei grafici a barre
    pa_df.plot(kind='bar', x='Nodes', y='Preferential Attachment Index', figsize=(8,6), rot=0)
    plt.title('Preferential Attachment Index')
    plt.xlabel('Nodes')
    plt.ylabel('Score')
    plt.show()

    # calcola il common neighbours index per tutte le coppie di vertici
    cn_scores = [(f"{i}-{j}", len(set(g.neighbors(i)) & set(g.neighbors(j)))) for i in range(g.vcount()) for j in range(g.vcount()) if i != j]
    # seleziona solo i 5 punteggi migliori
    top_5_cn = sorted(cn_scores, key=lambda x: x[1], reverse=True)[:5]
    cn_df = pd.DataFrame(top_5_cn, columns=['Nodes', 'Common Neighbours Index'])
    cn_df.plot(kind='bar', x='Nodes', y='Common Neighbours Index', figsize=(8,6), rot=0)
    plt.title('Common Neighbours Index')
    plt.xlabel('Nodes')
    plt.ylabel('Score')
    plt.show()

    # calcola il resource allocation index per tutte le coppie di vertici
    ra_scores = []
    for i in range(g.vcount()):
        for j in range(g.vcount()):
            if i != j and not g.are_connected(i, j):
                common_neighbors = set(g.neighbors(i)) & set(g.neighbors(j))
                if len(common_neighbors) > 0:
                    ra_scores.append((f"{i}-{j}", sum(1/g.degree(k) for k in common_neighbors)))
    # seleziona solo i 5 punteggi migliori
    top_5_ra = sorted(ra_scores, key=lambda x: x[1], reverse=True)[:5]
    ra_df = pd.DataFrame(top_5_ra, columns=['Nodes', 'Resource Allocation Index'])

    # creazione del grafico a dispersione
    #ra_df.plot(kind='scatter', x='Nodes', y='Resource Allocation Index', s=ra_df['Resource Allocation Index']*100, alpha=0.5, figsize=(8,6))
    ra_df.plot(kind='bar', x='Nodes', y='Resource Allocation Index', figsize=(8,6), rot=0)
    plt.title('Resource Allocation Index')
    plt.xlabel('Nodo sorgente')
    plt.ylabel('Nodo destinazione')

    plt.show()  

def failures(graph):
    #num_nodes_to_remove = len(graph.nodes()) #togli tutti i nodi 
    num_nodes_to_remove = int(0.3 * len(graph.nodes()))
    x_values = []
    y_values = []
    for i in range(num_nodes_to_remove):
        graph.remove_node(random.choice(list(graph.nodes())))
        scc = max(nx.strongly_connected_components(graph), key=len) if graph else set()
        x_values.append(i+1)
        y_values.append(len(scc))
    plt.plot(x_values, y_values, label='Random failure')
    plt.xlabel('Number of nodes removed')
    plt.ylabel('Size of largest strongly connected component')
    plt.legend()
    plt.show()
    efficiency = 0
    for node in graph.nodes():
        shortest_paths = nx.shortest_path_length(graph, source=node)
        node_efficiency = sum([1/path for path in shortest_paths.values() if path != 0])/(len(graph.nodes())-1)
        efficiency += node_efficiency
    efficiency /= len(graph.nodes())*(len(graph.nodes())-1)
    print("Efficiency:", efficiency)

def attackOutDegree(graph):
    out_degree = nx.out_degree_centrality(graph)
    sorted_out_degree = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
    num_nodes_to_remove = int(len(sorted_out_degree))

    x_values = []
    y_values = []

    for i in range(num_nodes_to_remove):
        nodes_to_remove = [n[0] for n in sorted_out_degree[:i+1]]
        graph.remove_nodes_from(nodes_to_remove)
        scc = max(nx.strongly_connected_components(graph), key=len) if graph else set()
        x_values.append(i+1)
        y_values.append(len(scc))

    print("New dimension of the largest strongly connected component (attack on out degree): " + str(len(scc)))
    plt.plot(x_values, y_values, label='Out degree attack')
    plt.xlabel('Number of nodes removed')
    plt.ylabel('Size of largest strongly connected component')
    plt.legend()
    plt.show()

def attackInDegree(graph):
    in_degree = nx.in_degree_centrality(graph)
    sorted_in_degree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
    num_nodes_to_remove = int(len(sorted_in_degree))

    x_values = []
    y_values = []

    for i in range(num_nodes_to_remove):
        nodes_to_remove = [n[0] for n in sorted_in_degree[:i+1]]
        graph.remove_nodes_from(nodes_to_remove)
        scc = max(nx.strongly_connected_components(graph), key=len) if graph else set()
        x_values.append(i+1)
        y_values.append(len(scc))

    print("New dimension of the largest strongly connected component (attack on in degree): " + str(len(scc)))
    plt.plot(x_values, y_values, label='In degree attack')
    plt.xlabel('Number of nodes removed')
    plt.ylabel('Size of largest strongly connected component')
    plt.legend()
    plt.show()

def attackPageRank(graph):
    pagerank = nx.pagerank(graph)
    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    num_nodes_to_remove = int(len(sorted_pagerank))

    x_values = []
    y_values = []

    for i in range(num_nodes_to_remove):
        nodes_to_remove = [n[0] for n in sorted_pagerank[:i+1]]
        graph.remove_nodes_from(nodes_to_remove)
        scc = max(nx.strongly_connected_components(graph), key=len) if graph else set()
        x_values.append(i+1)
        y_values.append(len(scc))

    print("New dimension of the largest strongly connected component (attack on PageRank): " + str(len(scc)))
    plt.plot(x_values, y_values, label='PageRank attack')
    plt.xlabel('Number of nodes removed')
    plt.ylabel('Size of largest strongly connected component')
    plt.legend()
    plt.show()

def autHub(graph):
    # calcolo delle centralità di hub e di autorità
    hubs, authorities = nx.hits(graph)
    # stampa dei primi 5 nodi con la centralità di hub più alta
    top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_dict_hubs = {node: score for node, score in top_hubs}
    # creazione del grafico a barre
    fig, ax = plt.subplots()
    plt.title("Hubs")
    positions = [1, 2, 3, 4, 5]  # posizioni sull'asse x
    clr = ['SeaGreen', 'MediumSeaGreen', 'PaleGreen', 'springGreen', 'LightGreen']
    ax.bar(positions, top_5_dict_hubs.values(), width=0.5, color = clr)
    # personalizzazione dell'asse x
    ax.set_xticks(positions)
    ax.set_xticklabels(top_5_dict_hubs.keys())
    ax.tick_params(axis='x', which='both', length=0)
    
    plt.show()

    top_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_dict_authorities = {node: score for node, score in top_authorities}
    # creazione del grafico a barre
    fig, ax = plt.subplots()
    plt.title("Authorities")
    positions = [1, 2, 3, 4, 5]  # posizioni sull'asse x
    clr = ['SeaGreen', 'MediumSeaGreen', 'PaleGreen', 'springGreen', 'LightGreen']
    ax.bar(positions, top_5_dict_authorities.values(), width=0.5, color = clr)
    # personalizzazione dell'asse x
    ax.set_xticks(positions)
    ax.set_xticklabels(top_5_dict_authorities.keys())
    ax.tick_params(axis='x', which='both', length=0)
    
    plt.show()

def data():
    # Dati del grafico
    y = ['Net. Diameter', 'Avg. Path length', 'Avg. Clustering Coefficient', 'Graph Density', 'Efficieny']
    x = [9, 3.390, 0.210, 0.013, 0.00018]
    clr = ['red', 'blue', 'green', 'purple', 'yellow']
    # Creazione del grafico
    plt.barh(y, x, color=clr)
    # Aggiunta di titolo e label degli assi
    plt.title('Network properties')
    plt.xlabel('Values')
    plt.ylabel('Properties')

    # Visualizzazione del grafico
    plt.show()


def plot_attacks(graph):
    num_nodes_to_remove = len(graph.nodes())
    x_values_failures = []
    x_values_page = []

    y_values_failures = []
    y_values_attack_out = []
    y_values_attack_in = []
    y_values_attack_pagerank = []
    y_value_attack_betweenness = []

    graph_copy_failures = graph.copy()
    
    for i in range(num_nodes_to_remove):
        # Failures attack
        graph_copy_failures.remove_node(random.choice(list(graph_copy_failures.nodes())))
        scc = max(nx.strongly_connected_components(graph_copy_failures), key=len) if graph_copy_failures else set()
        y_values_failures.append(len(scc))
        x_values_failures.append(i+1)
        print(i)

    for i in range(num_nodes_to_remove):
        # Out degree attack
        graph_copy = graph.copy()
        out_degree = nx.out_degree_centrality(graph_copy)
        sorted_out_degree = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
        nodes_to_remove = [n[0] for n in sorted_out_degree[:i+1]]
        graph_copy.remove_nodes_from(nodes_to_remove)
        scc = max(nx.strongly_connected_components(graph_copy), key=len) if graph_copy else set()
        y_values_attack_out.append(len(scc))

        # In degree attack
        graph_copy = graph.copy()
        in_degree = nx.in_degree_centrality(graph_copy)
        sorted_in_degree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
        nodes_to_remove = [n[0] for n in sorted_in_degree[:i+1]]
        graph_copy.remove_nodes_from(nodes_to_remove)
        scc = max(nx.strongly_connected_components(graph_copy), key=len) if graph_copy else set()
        y_values_attack_in.append(len(scc))

        # PageRank attack
        graph_copy = graph.copy()
        pagerank = nx.pagerank(graph_copy)
        sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        nodes_to_remove = [n[0] for n in sorted_pagerank[:i+1]]
        graph_copy.remove_nodes_from(nodes_to_remove)
        scc = max(nx.strongly_connected_components(graph_copy), key=len) if graph_copy else set()
        y_values_attack_pagerank.append(len(scc))

        # Betweenness attack
        graph_copy = graph.copy()
        betweenness = nx.betweenness_centrality(graph_copy)
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        nodes_to_remove = [n[0] for n in sorted_betweenness[:i+1]]
        graph_copy.remove_nodes_from(nodes_to_remove)
        scc = max(nx.strongly_connected_components(graph_copy), key=len) if graph_copy else set()
        y_value_attack_betweenness.append(len(scc))

        x_values_page.append(i+1)

    plt.plot(x_values_failures, y_values_failures, label='Random failure')
    plt.plot(x_values_page, y_values_attack_out, label='Out degree attack')
    plt.plot(x_values_page, y_values_attack_in, label='In degree attack')
    plt.plot(x_values_page, y_values_attack_pagerank, label='PageRank attack')
    plt.plot(x_values_page, y_value_attack_betweenness, label='Betweenness attack')
    plt.xlabel('Number of nodes removed')
    plt.ylabel('Size of largest strongly connected component')
    plt.legend()
    plt.show()

def global_efficieny(graph):
    efficiency = 0
    for node in graph.nodes():
        shortest_paths = nx.shortest_path_length(graph, source=node)
        node_efficiency = sum([1/path for path in shortest_paths.values() if path != 0])/(len(graph.nodes())-1)
        efficiency += node_efficiency
    efficiency /= len(graph.nodes())*(len(graph.nodes())-1)
    print("Efficiency:", efficiency)

def closenness(graph):
    # Calculate the closeness centrality and closeness out centrality of each node
    closeness_centrality = nx.closeness_centrality(graph)
    G_reverse = graph.reverse()
    closeness_out_centrality = nx.closeness_centrality(G_reverse, wf_improved=False)

    # stampa dei primi 5 nodi con la centralità di hub più alta
    top_cls_in = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_dict_cls_in = {node: score for node, score in top_cls_in}
    # creazione del grafico a barre
    fig, ax = plt.subplots()
    plt.title("In-Closeness Centrality")
    positions = [1, 2, 3, 4, 5]  # posizioni sull'asse x
    clr = ['SeaGreen', 'MediumSeaGreen', 'MediumSeaGreen', 'MediumSeaGreen', 'MediumSeaGreen']
    ax.bar(positions, top_5_dict_cls_in.values(), width=0.5, color = clr)
    # personalizzazione dell'asse x
    ax.set_xticks(positions)
    ax.set_xticklabels(top_5_dict_cls_in.keys())
    ax.tick_params(axis='x', which='both', length=0)
    
    plt.show()

    # stampa dei primi 5 nodi con la centralità di hub più alta
    top_cls_out = sorted(closeness_out_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_dict_cls_out = {node: score for node, score in top_cls_out}
    # creazione del grafico a barre
    fig, ax = plt.subplots()
    plt.title("Out-Closeness Centrality")
    positions = [1, 2, 3, 4, 5]  # posizioni sull'asse x
    clr = ['SeaGreen', 'SeaGreen', 'SeaGreen', 'SeaGreen', 'SeaGreen']
    ax.bar(positions, top_5_dict_cls_out.values(), width=0.5, color = clr)
    # personalizzazione dell'asse x
    ax.set_xticks(positions)
    ax.set_xticklabels(top_5_dict_cls_out.keys())
    ax.tick_params(axis='x', which='both', length=0)
    
    plt.show()

    # Write the results to a CSV file
    with open('C:/Users/nicco/OneDrive/Documenti/GitHub/Political-Blog-2004-U.S.-Election-Analysis/dataset/centrality_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Closeness Centrality', 'Closeness Out Centrality'])
        for node in graph.nodes():
            writer.writerow([node, closeness_centrality[node], closeness_out_centrality[node]])

def main():
    edges = pd.read_csv("C:/Users/nicco/OneDrive/Documenti/GitHub/Political-Blog-2004-U.S.-Election-Analysis/dataset/edge_list.csv", sep = ";")
    graph = nx.from_pandas_edgelist(edges, source = 'Source', target = 'Target', create_using=nx.DiGraph())
    #global_efficieny(graph)
    #data()
    #closenness(graph)
    #write_largest_connected_component(graph, 'C:/Users/nicco/OneDrive/Documenti/GitHub/Political-Blog-2004-U.S.-Election-Analysis/dataset/largest_component.csv')
    #communities = defineCommunities(graph)
    #mod = modularity(graph, communities)
    #scc, wcc, n_scc, n_wcc = components(graph)
    #InScaleFree(graph) #da rivedere
    #OutScaleFree(graph) #da rivedere
    #plotInDegreeDistribution(graph)
    #plotOutDegreeDistribution(graph)
    #plotInDegreeCumulativeDistribution(graph)
    #plotOutDegreeCumulativeDistribution(graph)
    #attackOutDegree(graph.copy())
    #attackInDegree(graph.copy())
    #attackPageRank(graph.copy())
    #failures(graph.copy())
    #drawCoreDecomposition(graph)
    #linkPrediction(edges)
    #autHub(graph)
    plot_attacks(graph)
    #pageRank(graph) #top 5 nodes
    #degree_centrality = degreeCentrality(graph) #top 5 nodes
    #betweennessCentrality(graph) #top 5 nodes
    #outDegreeCentrality(graph) #top 5 nodes
    #inDegreeCentrality(graph) #top 5 nodes
    #closenessCentrality(graph) #top 5 nodes
    #degree_centrality_str = ", ".join([str(node) for node in degree_centrality])
    #assortativity_coefficient = nx.degree_assortativity_coefficient(graph) # Calculate the assortativity coefficient
    
    #print("Assortativity coefficient: ", assortativity_coefficient) #Se postivo rete correlata, se negativo rete non correlata
    #print("Top 5 nodes degree centrality: " + degree_centrality_str)
    #print(page_rank.values()) #capire perchè non va e perchè l'ho fatto
    #print("Communities:", communities)
    #print("Modularity:", mod)
    #print("Dimension of the largest strongly connected component: " + str(len(scc)))
    #print("Dimension of the largest weakly connected component: " + str(len(wcc)))
    #print("Number of all the strongly connected components: " + str(n_scc))
    #print("Number of all the weakly connected components:" + str(n_wcc))

main()