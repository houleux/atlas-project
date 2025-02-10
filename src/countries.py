import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nxcom 
import matplotlib.pyplot as plt
from networkx.algorithms.centrality import in_degree_centrality, out_degree_centrality
from unidecode import unidecode
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors


def Graph_Maker(df_var):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    index = 0
    G = nx.DiGraph()

    while index < 26:
        start_l = df_var[df_var['Ending letter'] == alphabet[index]]
        end_l = df_var[df_var['Starting letter'] == alphabet[index]]

        for start_country in start_l['Countries']:
            for end_country in end_l['Countries']:
                if start_country != end_country:
                    G.add_edge(start_country, end_country)

        index += 1

    return G


# Load and preprocess data
countries = pd.read_csv('../data/country.csv')
countries.rename(columns={'value': 'Countries'}, inplace=True)

countries['Starting letter'] = countries['Countries'].str.lower().str[0].apply(unidecode)
countries['Ending letter'] = countries['Countries'].str.lower().str[-1].apply(unidecode)

countries.drop('id', axis=1, inplace=True)

# Create the graph
G = Graph_Maker(countries)

in_centrality = nx.in_degree_centrality(G)
out_centrality = nx.out_degree_centrality(G)

top_indegree_nodes = sorted(in_centrality, key=in_centrality.get, reverse=True)[:5]
top_outdegree_nodes = sorted(out_centrality, key=out_centrality.get, reverse=True)[:5]

communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
print(len(communities))

closeness_centrality = nx.closeness_centrality(G)

top_close_nodes = sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)[:5]

print(top_close_nodes)

# Node sizes based on in-degree
in_degrees = dict(G.in_degree())  # Get in-degree of each node
max_in_degree = max(in_degrees.values()) if in_degrees else 1
node_sizes = [30 + (deg / max_in_degree) * 900 for deg in in_degrees.values()]

# Color nodes by in-degree
norm = mcolors.Normalize(vmin=min(in_degrees.values()), vmax=max_in_degree)
cmap = plt.cm.Oranges

pos = nx.kamada_kawai_layout(G)

node_colors = ['blue' if node in top_close_nodes else cmap(norm(deg)) for node, deg in in_degrees.items()]

fig, ax = plt.subplots(figsize=(14, 10))

# Set the background color to black
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Draw the graph using the Kamada-Kawai layout


# Draw nodes and edges
nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='#c9bfa1', ax=ax, connectionstyle='arc3,rad=0.2')
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=cmap,
    edgecolors='white',  # Contrast edge color for black background
    linewidths=0.5,
    alpha=0.8,
    ax=ax
)
nx.draw_networkx_labels(G, pos, font_size=5, font_color='#e6eded', ax=ax)  # Bright label color for visibility

ax.set_title("Countries Connection Graph", fontsize=16, color='white')  # White title color
ax.axis('off')  # Turn off axis
plt.show()

nx.write_gexf(G, 'countries.gexf')
