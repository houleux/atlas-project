import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nxcom 
import matplotlib.pyplot as plt
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

        for start_city in start_l['city']:
            for end_city in end_l['city']:
                if start_city != end_city:
                    G.add_edge(start_city, end_city)

        index += 1

    return G


cot = pd.read_csv('cities.csv')

cities = cot.loc[:,['city']]

cities['Starting letter'] = cities['city'].str.lower().str[0].apply(unidecode)
cities['Ending letter'] = cities['city'].str.lower().str[-1].apply(unidecode)

G = Graph_Maker(cities)

# Node sizes based on in-degree
in_degrees = dict(G.in_degree())  # Get in-degree of each node
max_in_degree = max(in_degrees.values()) if in_degrees else 1
node_sizes = [30 + (deg / max_in_degree) * 900 for deg in in_degrees.values()]

# Color nodes by in-degree
norm = mcolors.Normalize(vmin=min(in_degrees.values()), vmax=max_in_degree)
cmap = plt.cm.Greens
node_colors = [cmap(norm(deg)) for deg in in_degrees.values()]

fig, ax = plt.subplots(figsize=(14, 10))

# Draw the graph using the Kamada-Kawai layout
pos = nx.kamada_kawai_layout(G)

# Draw nodes and edges
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax, connectionstyle='arc3,rad=0.2')
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=cmap,
    edgecolors='black',
    linewidths=0.5,
    ax=ax
)
nx.draw_networkx_labels(G, pos, font_size=5, font_color='#00b38f', ax=ax)

ax.set_title("City Connection Graph", fontsize=16)
ax.axis('off')  # Turn off axis
plt.show()
