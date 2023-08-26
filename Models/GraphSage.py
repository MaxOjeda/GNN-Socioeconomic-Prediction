import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv


class GraphSageRegression(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers=2, dropout=0.5):
        super(GraphSageRegression, self).__init__()
        self.drop = dropout
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.conv2 = SAGEConv(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv2.reset_parameters()


class GraphSAGEClassfication(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers=2, dropout=0.5):
        super(GraphSAGEClassfication).__init__()
        self.drop = dropout
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, output_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__