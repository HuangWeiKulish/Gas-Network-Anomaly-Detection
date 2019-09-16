import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

#gcn_msg = dgl.function.copy_src(src='h', out='m')  # message


def gcn_msg(edges):
    return {'m': edges.src['h']}


def gcn_reduce(nodes):
    # Todo: change it to weighted average, based on edge values (train it?)
    return {'h': torch.mean(nodes.mailbox['m'], dim=1)}


class TGCNCell(nn.Module):  # Graph Convolution Layer

    def __init__(self, in_feats, out_feats, tempo_nlayers=2, dropout=0.1, activf=F.relu, bias=True):
        super(TGCNCell, self).__init__()
        self.temporal_cell = nn.LSTM(input_size=in_feats, hidden_size=out_feats, num_layers=tempo_nlayers, bias=bias,
                                     dropout=dropout, batch_first=True)  # set bach_first=True so that input format is (batch, seq_len, input_size)
        self.dropout = dropout  # outputs are scaled by (1/(1-dropout)), default value in F.dropout is 0.5
        self.activf = activf

    def forward(self, g, inputs):
        """
        nodes send information computed via the message functions, and aggregates incoming information with the
        reduce functions
        :param g: graph
        :param inputs: inputs with dimension = (number_nodes, number_samples, sequence_length, number_features)
        :return:
        """
        number_nodes, number_samples, sequence_length, number_features = inputs.shape
        inputs = F.dropout(inputs, self.dropout, self.training)  # self.training is boolean, True while training

        # ----------------- aggregation from neighboring nodes -------------------------
        # set the node features
        g.ndata['h'] = inputs
        # trigger message passing on all edges
        g.send(g.edges(), gcn_msg)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')  # return value of key 'h' and remove this item from dictionary g.ndata

        # ----------------- pass to temporal nn -----------------
        out = torch.stack([self.temporal_cell(h[i, :, :, :])[0] for i in range(number_nodes)])  # stack output from each node
        # print(out_list.shape)
        out = self.activf(out)
        return out


class TGCNAE(nn.Module):  # Graph Auto-Encoder

    def __init__(self, in_feats, tgcnout_feats, hidden_dim_list, dropout, activf=F.relu):
        """
        This is a symmetric auto-encoder
        :param in_feats:
        :param hidden_dim_list: a list where each element is the dimension of a hidden layer
        :param dropout:
        """
        super(TGCNAE, self).__init__()
        self.in_feats, self.hidden_dim_list, self.dropout, self.activf = in_feats, hidden_dim_list, dropout, activf
        self.tgcn_layer = TGCNCell(in_feats, tgcnout_feats, dropout, activf)
        self.encoding_dims = [tgcnout_feats] + hidden_dim_list
        self.decoding_dims = self.encoding_dims[::-1]
        self.encoder = self.encode()
        self.decoder = self.decode()

    def encode(self):
        layers = []
        n_layers = len(self.encoding_dims)
        for n in range(1, n_layers):
            layers.append(nn.Linear(in_features=self.encoding_dims[n-1], out_features=self.encoding_dims[n], bias=True))
            layers.append(nn.ELU(alpha=1.0, inplace=False))
        return nn.Sequential(*layers)

    def decode(self):
        layers = []
        n_layers = len(self.decoding_dims)
        for n in range(1, n_layers):
            layers.append(nn.Linear(in_features=self.decoding_dims[n-1], out_features=self.decoding_dims[n], bias=True))
            layers.append(nn.ELU(alpha=1.0, inplace=False))
        return nn.Sequential(*layers)

    def forward(self, g, inputs):
        x = self.tgcn_layer(g, inputs)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
in_feats = 3  # 2 time features and 1 pressure features
tgcnout_feats = 1  # output 1 feature from tgcn cell
hidden_dim_list = [20, 10]
dropout = 0.1
activf = F.relu

gasnetwork_model = TGCNAE(in_feats, tgcnout_feats, hidden_dim_list, dropout, activf)
print(gasnetwork_model)









