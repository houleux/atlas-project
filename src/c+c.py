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

        for start_place in start_l['place']:
            for end_place in end_l['place']:
                if start_place != end_place:
                    G.add_edge(start_place, end_place)

        index += 1

    return G


cot = pd.read_csv('../data/cities.csv')
countries = pd.read_csv('../data/country.csv')

cities = cot.loc[:, ['city']]
cities.rename(columns={'city':'place'}, inplace=True)
cities['place'] = cities['place'].str.strip().str.lower().apply(unidecode)
cities['Starting letter'] = cities['place'].str[0]
cities['Ending letter'] = cities['place'].str[-1]

countries.rename(columns={'value': 'place'}, inplace=True)
countries['place'] = countries['place'].str.strip().str.lower().apply(unidecode)
countries['Starting letter'] = countries['place'].str[0]
countries['Ending letter'] = countries['place'].str[-1]

countries.drop('id', axis=1, inplace=True)

places = pd.merge(cities.head(500), countries, on=['place', 'Starting letter', 'Ending letter'], how='outer')

G = Graph_Maker(places)


# Node sizes based on in-degree
in_degrees = dict(G.in_degree())  # Get in-degree of each node
max_in_degree = max(in_degrees.values()) if in_degrees else 1
node_sizes = [30 + (deg / max_in_degree) * 900 for deg in in_degrees.values()]

# Color nodes by in-degree
norm = mcolors.Normalize(vmin=min(in_degrees.values()), vmax=max_in_degree)
cmap = plt.cm.Blues
node_colors = [cmap(norm(deg)) for deg in in_degrees.values()]

fig, ax = plt.subplots(figsize=(14, 10))

# Set the background color to black
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Draw the graph using the Kamada-Kawai layout
pos = nx.kamada_kawai_layout(G)

# Draw nodes and edges
nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='#495b5c', ax=ax, connectionstyle='arc3,rad=0.2')
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

ax.set_title("City+Countries Connection Graph", fontsize=16, color='white')  # White title color
ax.axis('off')  # Turn off axis
plt.show()



nx.write_gexf(G, 'places.gexf')
