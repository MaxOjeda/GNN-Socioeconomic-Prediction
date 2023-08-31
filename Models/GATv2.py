import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATv2(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, edge_dim=6, heads=1, dropout=0.5, num_layers=2):
        super(GATv2, self).__init__()
        self.dropout = dropout
        self.heads = heads
        self.edge_dim = edge_dim
        
        self.conv1 = GATv2Conv(input_channels, hidden_channels, heads=self.heads, dropout=self.dropout, edge_dim=self.edge_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels*self.heads, hidden_channels*self.heads))
        self.conv2 = GATv2Conv(hidden_channels*self.heads, output_channels, concat=False, dropout=self.dropout)

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