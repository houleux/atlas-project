from torch.optim import optimizer
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from sklearn.metrics import roc_auc_score
import torch

import torch.nn.functional as F 

from gensim.models import Word2Vec
import networkx as nx
from unidecode import unidecode

import matplotlib as plt
from torch_geometric.datasets import Planetoid

import numpy as np
import pandas as pd


def Graph_Maker(df_var):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    index = 0
    G = nx.DiGraph()

    while index < 26:
        start_l = df_var[df_var['Ending letter'] == alphabet[index]]
        end_l = df_var[df_var['Starting letter'] == alphabet[index]]

        G.add_nodes_from(start_l['Countries'].tolist(), e_letter = index)
        G.add_nodes_from(end_l['Countries'].tolist(), s_letter = index)

        for start_country in start_l['Countries']:
            for end_country in end_l['Countries']:
                if start_country != end_country:
                    G.add_edge(start_country, end_country)


        index += 1

    return G


countries = pd.read_csv('../data/country.csv')
countries.rename(columns={'value': 'Countries'}, inplace=True)

countries['Starting letter'] = countries['Countries'].str.lower().str[0].apply(unidecode)
countries['Ending letter'] = countries['Countries'].str.lower().str[-1].apply(unidecode)

countries.drop('id', axis=1, inplace=True)

G = Graph_Maker(countries)

G.nodes['Myanmar (Burma)']['e_letter'] = 17

########################################################

node_map = {node: i for i, node in enumerate(G.nodes())}

edge_index = torch.tensor([(node_map[u], node_map[v]) for u, v in G.edges]).t().contiguous()
num_nodes = G.number_of_nodes()
features = torch.tensor([[G.nodes[node]["s_letter"], G.nodes[node]["e_letter"]] for node in G.nodes()])


data = Data(edge_index=edge_index, num_nodes=num_nodes, x=features)


node2vec = Node2Vec(edge_index, embedding_dim=32, walk_length=10, context_size=5, walks_per_node=10)

optimizer = torch.optim.Adam(node2vec.parameters(), lr = 0.01)


def train_nodey():
    node2vec.train()
    total_l = 0

    for i in range(500):
        optimizer.zero_grad()
        batch = torch.randint(0, data.num_nodes, (data.num_nodes,))
        pos_rw, neg_rw = node2vec.sample(batch=batch)

        loss = node2vec.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()


train_nodey()

## final loss: 4->1

node_embeddings = node2vec.forward()

positive_edges = edge_index.t()

# Generate random negative edges
negative_edges = []
while len(negative_edges) < positive_edges.shape[0]:
    u = torch.randint(0, num_nodes, (1,)).item()
    v = torch.randint(0, num_nodes, (1,)).item()
    if not G.has_edge(list(node_map.keys())[u], list(node_map.keys())[v]) and u != v:
        negative_edges.append((u, v))


negative_edges = torch.tensor(negative_edges)

def extract_features(edges):
    src, dst = edges.t()
    src_emb = node_embeddings[src]
    dst_emb = node_embeddings[dst]
    return torch.cat([src_emb, dst_emb], dim=-1)


x_pos = extract_features(positive_edges)
x_neg = extract_features(negative_edges)

y_pos = torch.ones(x_pos.shape[0])
y_neg = torch.zeros(x_neg.shape[0])

x_train = torch.cat([x_pos, x_neg], dim=0)
y_train = torch.cat([y_pos, y_neg], dim=0)


class LinkPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 16)
        self.fc2 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # No sigmoid here


model = LinkPredictor(64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()


def train_model():
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()

        # Detach input features each epoch to prevent graph retention
        x_train_epoch = x_train.detach()

        y_pred = model(x_train_epoch).squeeze()
        loss = loss_fn(y_pred, y_train)

        # Detach computational graph properly
        loss.backward(retain_graph=False)

        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
train_model()

model.eval()
with torch.no_grad():
    y_pred = model(x_train).squeeze()
    auc = roc_auc_score(y_train.numpy(), y_pred.numpy())
    print(f"ROC AUC: {auc:.4f}")
