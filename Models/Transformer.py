import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import TransformerConv

class TransformerClassification(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, edge_dim=2, heads=1, dropout=0.5, num_layers=1):
        super(TransformerClassification, self).__init__()
        self.dropout = dropout
        self.heads = heads
        self.edge_dim = edge_dim
        
        self.conv1 = TransformerConv(input_channels, hidden_channels, heads=self.heads, dropout=self.dropout, edge_dim=self.edge_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden_channels*self.heads, hidden_channels*self.heads))
        self.conv2 = TransformerConv(hidden_channels*self.heads, output_channels, concat=False, dropout=self.dropout)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.edge_dim != None:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = F.elu(x)
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    

class TransformerRegression(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, edge_dim=2, heads=1, dropout=0.5, num_layers=1):
        super(TransformerRegression, self).__init__()
        self.dropout = dropout
        self.heads = heads
        self.edge_dim = edge_dim
        
        self.conv1 = TransformerConv(input_channels, hidden_channels, heads=self.heads, dropout=self.dropout, edge_dim=self.edge_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden_channels*self.heads, hidden_channels*self.heads))
        self.conv2 = TransformerConv(hidden_channels*self.heads, output_channels, concat=False, dropout=self.dropout)
        #self.linear = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.edge_dim != None:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = F.elu(x)
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x