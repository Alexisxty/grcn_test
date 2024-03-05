import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import pytorch_lightning as pl
#from components.GRCN_components import cal_similarity_graph, create_similarity_graph, _node_embeddings
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv

class GraphConvolution(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # Apply linear transformation
        h = self.linear(inputs)
        # Aggregate neighbor features
        g.ndata['h'] = h
        g.update_all(dgl.function.copy_u('h', 'm'),
                     dgl.function.sum('m', 'h'))
        h = g.ndata.pop('h')
        # Apply non-linearity
        return F.relu(h)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(in_feats, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = self.gcn2(g, h)
        return h


class GCNConvDGL(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConvDGL, self).__init__()
        # 使用DGL的GraphConv
        self.graph_conv = GraphConv(in_feats=input_size, out_feats=output_size,
                                    norm='none', weight=True, bias=True)

    def forward(self, g, input_features):
        '''
        g: DGL图
        input_features: 节点的特征矩阵
        '''
        # DGL图卷积操作
        output_features = self.graph_conv(g, input_features)
        return output_features


class DiagDGL(nn.Module):
    def __init__(self, input_size, device):
        super(DiagDGL, self).__init__()
        self.W = nn.Parameter(torch.ones(input_size, device=device))
        self.device = device

    def forward(self, g, features):
        hidden = features * self.W
        return hidden


class GCNConvDiagDGL(nn.Module):
    def __init__(self, input_size, device):
        super(GCNConvDiagDGL, self).__init__()
        self.W = nn.Parameter(torch.ones(input_size, device=device))
        self.device = device

    def forward(self, g, features):
        weighted_features = features * self.W
        g.ndata['h'] = weighted_features
        g.update_all(message_func=dgl.function.copy_u('h', 'm'),
                     reduce_func=dgl.function.sum('m', 'h'))
        output = g.ndata.pop('h')

        return output



def create_similarity_graph(g, similarity_matrix, norm_mode='both'):
    adj_matrix = g.adjacency_matrix(scipy_fmt='coo').to_dense()
    new_adj_matrix = adj_matrix + similarity_matrix
    # 归一化处理
    if norm_mode == 'both':
        # 对行和列进行归一化
        rowsum = new_adj_matrix.sum(dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        new_adj_matrix = d_mat_inv_sqrt @ new_adj_matrix @ d_mat_inv_sqrt
    elif norm_mode == 'right':
        # 对列进行归一化
        rowsum = new_adj_matrix.sum(dim=1)
        d_inv = torch.pow(rowsum, -1).flatten()
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diag(d_inv)
        new_adj_matrix = new_adj_matrix @ d_mat_inv
    elif norm_mode == 'diag':
        # 对角线归一化：使每个节点的所有邻接权重之和为1
        rowsum = new_adj_matrix.sum(dim=1)
        d_inv = torch.pow(rowsum, -1).flatten()
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diag(d_inv)
        new_adj_matrix = d_mat_inv @ new_adj_matrix
    # 使用归一化后的邻接矩阵创建新的DGL图
    src, dst = new_adj_matrix.nonzero()
    weights = new_adj_matrix[src, dst]
    new_g = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
    new_g.edata['weight'] = weights.float()

    return new_g

def _node_embeddings(self, g, inputs):
    h = self.conv1(g, inputs)
    h = F.relu(h)
    h = self.conv2(g, h)
    if self.normalize:
        h = F.normalize(h, p=2, dim=1)
    return h

def cal_similarity_graph(self, node_embeddings):
    similarity_graph = torch.mm(node_embeddings[:, :int(self.num_features / 2)],
                                node_embeddings[:, :int(self.num_features / 2)].t())
    similarity_graph += torch.mm(node_embeddings[:, int(self.num_features / 2):],
                                node_embeddings[:, int(self.num_features / 2):].t())
    return similarity_graph

###############################################components###############################################

class GRCNModel(pl.LightningModule):
    def __init__(self, num_features, num_classes, hid_feats, graph_nhid, graph_nhid2, dropout=0.5, lr=1e-3):
        super(GRCNModel, self).__init__()
        self.save_hyperparameters()
        self.conv_graph1 = GraphConv(num_features, graph_nhid, activation=F.relu)
        self.conv_graph2 = GraphConv(graph_nhid, graph_nhid2, activation=None)
        self.conv1 = GraphConv(num_features, hid_feats, activation=F.relu)
        self.conv2 = GraphConv(hid_feats, num_classes, activation=None)
        self.F_graph = F.relu
        self.dropout = dropout
        self.accuracy = Accuracy(num_classes=num_classes, average='macro', task="multiclass")

        self.criterion = nn.CrossEntropyLoss()



    def forward(self, g, features):
        # 图修订部分
        node_embeddings = _node_embeddings(g, features)
        similarity_graph = cal_similarity_graph(node_embeddings)
        new_g = create_similarity_graph(g, similarity_graph)
        # 用于图分类的特征提取
        x = self.conv1(new_g, features)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(new_g, x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        g, features, labels = batch
        logits = self(g, features)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)


if __name__ == "__main__":
    _ = GRCNModel()
