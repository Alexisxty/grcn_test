import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GRCNModel(nn.Module):
    def __init__(self, num_features, num_classes, hid_feats, graph_nhid, graph_nhid2, dropout=0.5):
        super(GRCNModel, self).__init__()
        self.conv_graph1 = GraphConv(num_features, graph_nhid, activation=F.relu)
        self.conv_graph2 = GraphConv(graph_nhid, graph_nhid2, activation=None)
        self.conv1 = GraphConv(num_features, hid_feats, activation=F.relu)
        self.conv2 = GraphConv(hid_feats, num_classes, activation=None)
        self.F_graph = F.relu
        self.dropout = dropout
        self.num_features = num_features
        self.normalize = True



    def _node_embeddings(self, g, inputs):
        # 应用第一个图卷积层
        h = self.conv1(g, inputs)
        # 应用ReLU激活函数
        h = F.relu(h)
        # 应用第二个图卷积层
        h = self.conv2(g, h)
        # 可选的L2归一化
        if self.normalize:
            h = F.normalize(h, p=2, dim=1)
        return h

    def cal_similarity_graph(self, node_embeddings):
        # similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
        similarity_graph = torch.mm(node_embeddings[:, :int(self.num_features / 2)],
                                    node_embeddings[:, :int(self.num_features / 2)].t())
        similarity_graph += torch.mm(node_embeddings[:, int(self.num_features / 2):],
                                     node_embeddings[:, int(self.num_features / 2):].t())
        return similarity_graph

    def create_similarity_graph(self, g, similarity_matrix, norm_mode='both'):
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

    def forward(self, g, features):
        # 图修订部分
        node_embeddings = self._node_embeddings(g, features)
        similarity_graph = self.cal_similarity_graph(node_embeddings)
        new_g = self.create_similarity_graph(g, similarity_graph)
        # 用于图分类的特征提取
        x = self.conv1(new_g, features)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(new_g, x)
        return F.log_softmax(x, dim=1)


import pytorch_lightning as pl
from torchmetrics import Accuracy


class GRCNLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(GRCNLightningModule, self).__init__()
        self.save_hyperparameters(config)
        self.model = GRCNModel(**config.model)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, g, features):
        return self.model(g, features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.config.optimizer.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        g, features, labels = batch
        logits = self.forward(g, features)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        g, features, labels = batch
        logits = self.forward(g, features)
        loss = self.criterion(logits, labels)
        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)
        self.accuracy.update(preds, labels)
        self.log('val_acc', self.accuracy.compute(), prog_bar=True)

if __name__ == "__main__":
    _ = GRCNLightningModule()