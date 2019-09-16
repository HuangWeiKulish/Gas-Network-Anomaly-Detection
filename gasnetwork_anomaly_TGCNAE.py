import itertools
import datetime
import pandas as pd
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
import numpy as np
import pickle as pkl
import os
import networkx as nx
import torch as th
import torch.nn.functional as F
import dgl

import sys
sys.path.append(os.path.join(os.getcwd(), 'GasNetwork'))
from Graph_AutoEncoder import TGCNAE


# ---------------- Prepare Data -----------------
data = pkl.load(open(os.path.join(os.getcwd(), 'GasNetwork', 'simulated_data.pkl'), "rb"))
maincols = ['pressure_abnormal', 'anomly_flag']
combined_df = pd.concat([data[k][maincols].rename(columns={c: k+'_'+c for c in maincols}) for k in data.keys()], axis=1)
combined_df['weekday_value'] = combined_df.index.weekday / 6
combined_df['time_value'] = (combined_df.index.hour*60+combined_df.index.minute) / (23*60+59)

print(combined_df.shape, combined_df.dtypes)
print(combined_df.head())

pressure_cols = [c for c in list(combined_df) if 'pressure' in c]
time_cols = ['weekday_value', 'time_value']
label_cols = [c for c in list(combined_df) if 'flag' in c]
n_nodes = len(pressure_cols)


# ---------------- Build Graph ----------------
def create_edges_fully_connected(n_nodes):
    edges = list(itertools.permutations(np.arange(n_nodes), 2))  # permutation makes the edges bi-directional
    src, dst = tuple(zip(*edges))
    return src, dst

gasnetwork_graph = dgl.DGLGraph()
gasnetwork_graph.add_nodes(n_nodes)  # each sensor is a node, node label is from 0
src, dst = create_edges_fully_connected(n_nodes)  # build a fully connected graph, because gas network is assumed to be spider net like
gasnetwork_graph.add_edges(src, dst)
print('number of nodes = {}'.format(gasnetwork_graph.number_of_nodes()))
print('number of edges = {}'.format(gasnetwork_graph.number_of_edges()))
gasnetwork_graph.set_n_initializer(dgl.init.zero_initializer)  # Set the initializer for empty node features. (Set to 0)
gasnetwork_graph.set_e_initializer(dgl.init.zero_initializer)  # Set the initializer for empty edge features. (Set to 0)


# visualize the graph
gnw_g = gasnetwork_graph.to_networkx().to_undirected()
gnw_pos = nx.kamada_kawai_layout(gnw_g)  # Kamada-Kawaii layout
nx.draw(gnw_g, gnw_pos, with_labels=True, node_color=[[0.5, 0.6, 0.3]])
# plt.show()

# assign data to each nodes -----------------

def chopts(df, length, nodes_cols_dict):
    n_nodes = len(nodes_cols_dict)
    new_df = {k: {'val': [], 'cols': nodes_cols_dict[k]} for k in range(len(nodes_cols_dict))}
    for i in range(len(df)-length+1):
        tmp = df.iloc[i:length+i]
        for j in range(n_nodes):
            new_df[j]['val'].append(tmp[nodes_cols_dict[j]].values)
    for k in new_df.keys():
        new_df[k]['val'] = np.stack(new_df[k]['val'])
    return new_df


gap = datetime.timedelta(seconds=60*10)
period = int(datetime.timedelta(days=1) / gap)
nodes_cols_dict = [[c]+time_cols for c in pressure_cols]
new_combined_df = chopts(combined_df, period, nodes_cols_dict)

g_data = [new_combined_df[k]['val'] for k in new_combined_df.keys()]
g_data = np.stack(g_data)
print(g_data.shape)

gasnetwork_graph




# ---------------- Train Model ----------------

in_feats = 3  # 2 time features and 1 pressure features
tgcnout_feats = 1  # output 1 feature from tgcn cell
seq_length = 144
hidden_dim_list = [20, 10]
dropout = 0.1
activf = F.relu
inputs = th.tensor(g_data, dtype=th.float32)


gasnetwork_model = TGCNAE(in_feats, tgcnout_feats, seq_length, hidden_dim_list,
                          tempo_nlayers=2, dropout=0.1, activf=F.relu, bias=True)
optimizer = th.optim.Adam(gasnetwork_model.parameters(), lr=0.01)


all_logits = []
for epoch in range(30):
    logits = gasnetwork_model(gasnetwork_graph, inputs)
    # we save the logits for visualization later
    all_logits.append(logits.detach())  # .detach is to get the data
    loss = F.mse_loss(logits, inputs[:, :, :, 0].unsqueeze(3))  # .unsqueeze to extend dimension

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))


