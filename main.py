import pandas as pd
import csv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import infomap

from igraph import *
import networkit as nt
import matplotlib.colors as colors
import matplotlib.cm as cm

from pathlib import Path
from collections import Counter
from networkx.algorithms.distance_measures import diameter

'''
input_file = "C:/Users/nicco/OneDrive/Documenti/ProgettoCSR/orientation.csv"
output_file = "C:/Users/nicco/OneDrive/Documenti/ProgettoCSR/orientation_index.csv"

# Open the input and output CSV files
with open(input_file, "r") as input_f, open(output_file, "w", newline="") as output_f:
    reader = csv.reader(input_f)
    writer = csv.writer(output_f)
    
    # Iterate over the input rows and add an index to each row
    for i, row in enumerate(reader):
        row_with_index = [i] + row 
        writer.writerow(row_with_index)

# Print a message to confirm that the output file was created
print(f"Indexed CSV file created: {output_file}")


edge = pd.read_csv("C:/Users/nicco/OneDrive/Documenti/ProgettoCSR/edge_list.csv")
url = pd.read_csv("C:/Users/nicco/OneDrive/Documenti/ProgettoCSR/orientation_index.csv", usecols=["Orientation"])


def get_url(row):
    f, t = row["Source;Target"].split(";")
    return {"Source": url['Orientation'].iloc[int(f) - 1], 'Target': url['Orientation'].iloc[int(t) - 1]}

merged = edge.apply(get_url, axis=1, result_type='expand')
merged.columns = ["Source", "Target"]
merged.to_csv("C:/Users/nicco/OneDrive/Documenti/ProgettoCSR/Blog_link_gephi.csv", index = False)

def createGraph():
    with open('C:/Users/nicol/Desktop/darkweb.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        edges = []
        next(reader)
        for row in reader:
            edges.append((row[0], row[1], float(row[3])))

    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    return graph

//////
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
    print("Dimension of the largest strongly connected component: " + str(len(scc)))
    wcc = max(nx.weakly_connected_components(graph), key=len)
    print("Dimension of the largest weakly connected component: " + str(len(wcc)))
    n_scc = nx.number_strongly_connected_components(graph)
    print("Number of all the strongly connected components: " + str(n_scc))
    n_wcc = nx.number_weakly_connected_components(graph)
    print("Number of all the weakly connected components:" + str(n_wcc))

def degreeCentrality(graph):
    degree = nx.degree_centrality(graph)
    sorted_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_degree[:5]
    return top_5

def outDegreeCentrality(graph):
    out_degree = nx.out_degree_centrality(graph)
    sorted_out_degree = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_out_degree[:5]
    return top_5

def inDegreeCentrality(graph):
    in_degree = nx.in_degree_centrality(graph)
    sorted_in_degree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_in_degree[:5]
    return top_5

def betweennessCentrality(graph):
    betweenness = nx.betweenness_centrality(graph)
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_betweenness[:5]
    return top_5

def closenessCentrality(graph):
    closeness = nx.closeness_centrality(graph)
    sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_closeness[:5]
    return top_5

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
    plt.ylabel("P_in(k)")
    plt.plot(deg, cnt)
    plt.show()
    plt.title("In-Degree cumulated distribution")

def plotOutDegreeDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.out_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    plt.title("Out-Degree distribution")
    plt.xlabel("k")
    plt.ylabel("P_out(k)")
    plt.plot(deg, cnt)
    plt.show()

def plotInDegreeCumulativeDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.in_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    cdf = np.cumsum(cnt)
    
    plt.figure(figsize=(12, 8))
    plt.title("In-Degree cumulated distribution")
    plt.xlabel("k")
    plt.ylabel("P_in_cumulative(k)")
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
    
    plt.figure(figsize=(12, 8))
    plt.title("Out-Degree cumulated distribution")
    plt.xlabel("k")
    plt.ylabel("P_out_cumulative(k)")
    plt.plot(deg, cdf, '-o', markersize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def InScaleFree(graph):
    # Calcolo la distribuzione del grado dei nodi
    degree_dist = [d for n, d in graph.in_degree()]
    # Fitting della distribuzione con una legge di potenza
    fit = powerlaw.Fit(degree_dist, discrete=True)
    # Plot della distribuzione del grado e della legge di potenza fittata
    fig, ax = plt.subplots()
    fit.plot_pdf(color='b', linewidth=2)
    fit.power_law.plot_pdf(color='r', linestyle='--', ax=ax)
    plt.xlabel('In-Degree Distribution')
    plt.ylabel('In-degree Power Law Fit')
    plt.legend(['In-Degree Distribution', 'Power Law Fit'])
    plt.show()
    # Print the power-law exponent
    print(f"Power-law exponent for In-Degree Distribution: {fit.power_law.alpha}") #tra 2 e 3 è scale free, se no no

def OutScaleFree(graph):
    # Calcolo la distribuzione del grado dei nodi
    degree_dist = [d for n, d in graph.out_degree()]
    # Fitting della distribuzione con una legge di potenza
    fit = powerlaw.Fit(degree_dist, discrete=True)
    # Plot della distribuzione del grado e della legge di potenza fittata
    fig, ax = plt.subplots()
    fit.plot_pdf(color='b', linewidth=2)
    fit.power_law.plot_pdf(color='r', linestyle='--', ax=ax)
    plt.xlabel('Out-Degree Distribution')
    plt.ylabel('Out-degree Power Law Fit')
    plt.legend(['Out-Degree Distribution', 'Power Law Fit'])
    plt.show()
    # Print the power-law exponent
    print(f"Power-law exponent for Out-Degree Distribution: {fit.power_law.alpha}") #tra 2 e 3 è scale free, se no no

def drawUncorrelatedNetwroks(graph):
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, pos, node_size=15)
    nx.draw_networkx_edges(graph, pos, width=0.3, alpha = 0.5)
    plt.axis('off')
    plt.show() #nodi con un alto grado di entrata tendono a connettersi con nodi con un basso grado di uscita

def pageRank(graph):
    # Compute the PageRank score for each node
    pr = nx.pagerank(graph, alpha=0.85, tol = 0.001)
    # Get the top 5 nodes with the highest PageRank score
    top_nodes = sorted(pr, key=pr.get, reverse=True)[:5]

    for node in top_nodes:
        print(f"{node}: {pr[node]}")
    return top_nodes

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

def compute_infomap(graph):
    im = infomap.Infomap("--directed")
    print("---")
    for edge in graph.edges():
        print(edge[0], edge[1])
        im.addLink(edge[0], edge[1])
    im.run()
    communities = {}
    for node in im.iterTree():
        if node.isLeaf():
            community = node.moduleIndex()
            node_id = node.physicalId
            if community not in communities:
                communities[community] = []
            communities[community].append(node_id)
    print(communities)

def main():
    edges = pd.read_csv("C:/Users/nicco/OneDrive/Documenti/GitHub/Political-Blog-2004-U.S.-Election-Analysis/dataset/edge_list.csv", sep = ";")
    graph = nx.from_pandas_edgelist(edges, source = 'Source', target = 'Target', create_using=nx.DiGraph())
    #compute_infomap(graph)
    drawCoreDecomposition(graph)
    '''
    with open('C:/Users/nicco/OneDrive/Documenti/GitHub/Political-Blog-2004-U.S.-Election-Analysis/dataset/edge_list.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        edges = []
        next(reader)
        for row in reader:
            edges.append((row[0], row[1]))
        graph = nx.DiGraph()
        graph.add_edges_from(edges)

        # Definisci le comunità
        #communities = defineCommunities(graph)

        # Calcola la modularità
        #mod = modularity(graph, communities)

        components(graph)
        #InScaleFree(graph)
        #OutScaleFree(graph)
        #plotInDegreeDistribution(graph)
        #plotOutDegreeDistribution(graph)
        #plotInDegreeCumulativeDistribution(graph)
        #plotOutDegreeCumulativeDistribution(graph)
        #drawUncorrelatedNetwroks(graph)
        #kcore(graph)
        #compute_k_core_decomposition(graph)
        compute_infomap(graph)

        page_rank = pageRank(graph) #top 5 nodes
        degree_centrality = degreeCentrality(graph) #top 5 nodes
        betweenness_centrality = betweennessCentrality(graph) #top 5 nodes
        out_degree = outDegreeCentrality(graph) #top 5 nodes
        in_degree = inDegreeCentrality(graph) #top 5 nodes
        closeness_centrality = closenessCentrality(graph) #top 5 nodes
        degree_centrality_str = ", ".join([str(node) for node in degree_centrality])
        out_degree_str = ", ".join([str(node) for node in out_degree])
        in_degree_str = ", ".join([str(node) for node in in_degree])
        betweenness_centrality_str = ", ".join([str(node) for node in betweenness_centrality])
        closeness_centrality_str = ", ".join([str(node) for node in closeness_centrality])
        assortativity_coefficient = nx.degree_assortativity_coefficient(graph) # Calculate the assortativity coefficient

        print("Assortativity coefficient: ", assortativity_coefficient) #Se postivo rete correlata, se negativo rete non correlata
        print("Top 5 nodes degree centrality: " + degree_centrality_str)
        print("Top 5 nodes out degree: " + out_degree_str)
        print("Top 5 nodes in degree: " + in_degree_str)
        print("Top 5 nodes betweenness centrality: " + betweenness_centrality_str)
        print("Top 5 nodes closeness centrality: " + closeness_centrality_str)
        #print(page_rank.values())
        #print("Communities:", communities)
        #print("Modularity:", mod)
    '''
main()