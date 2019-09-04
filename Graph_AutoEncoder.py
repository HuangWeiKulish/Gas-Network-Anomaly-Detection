import dgl.function as dfn
import torch.nn as nn
import torch.nn.functional as F


gcn_msg = dfn.copy_src(src='h', out='m')  # message
gcn_reduce = dfn.sum(msg='m', out='h')  # reduce function


class GCNLayer(nn.Module):  # Graph Convolution Layer

    def __init__(self, in_feats, out_feats, dropout=0.0, activf=F.relu):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.dropout = dropout  # outputs are scaled by (1/(1-dropout)), default value in F.dropout is 0.5
        self.activf = activf

    def forward(self, g, inputs):
        """
        nodes send information computed via the message functions, and aggregates incoming information with the
        reduce functions
        :param g: graph
        :param inputs: input node features
        :return:
        """
        inputs = F.dropout(inputs, self.dropout, self.training)  # self.training is boolean, True while training
        # set the node features
        g.ndata['h'] = inputs
        # trigger message passing on all edges
        g.send(g.edges(), gcn_msg)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')  # return value of key 'h' and remove this item from dictionary g.ndata
        # perform linear transformation
        out = self.linear(h)
        # activate
        out = self.activf(out)
        return out


class GCNAE(nn.Module):  # Graph Auto-Encoder

    def __init__(self, in_feats, gcn_out, hidden_dim_list, dropout, activf=F.relu):
        """
        This is a symmetric auto-encoder
        :param in_feats:
        :param hidden_dim_list: a list where each element is the dimension of a hidden layer
        :param dropout:
        """
        super(GCNAE, self).__init__()
        self.in_feats, self.hidden_dim_list, self.dropout, self.activf = in_feats, hidden_dim_list, dropout, activf
        self.gcn_layer = GCNLayer(in_feats, gcn_out, dropout, activf)
        self.encoding_dims = [gcn_out] + hidden_dim_list
        self.decoding_dims = self.encoding_dims[::-1]
        self.encoder = self.encode()
        self.decoder = self.decode()

    def encode(self):
        layers = []
        n_layers = len(self.encoding_dims)
        for n in range(1, n_layers):
            layers.append(nn.LSTM(input_size=self.in_feats, hidden_size=10, num_layers=1))
        return nn.Sequential(*layers)

    def decode(self):
        layers = []
        n_layers = len(self.decoding_dims)
        for n in range(1, n_layers):
            layers.append(nn.LSTM(self.decoding_dims[n-1], self.decoding_dims[n], self.dropout, self.activf))
        return nn.Sequential(*layers)

    def forward(self, g, input):
        x = self.gcn_layer(g, input)
        x = self.encoder(x)
        x = self.decoder(x)
        return x





