import torch
import torch_sparse
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from Models.Feature_Propagation import FeaturePropagation


def createPytorchData(G, file_name: str):
    # 1. Cast a flotantes
    for att in ['beautiful', 'boring', 'depressing', 'lively', 'safe', 'wealthy', 'bc', 'eigen', 'pg_rank', 'division', 'ismt',
                'houses', 'hog_40pct', 'pct_hog40p', 'ave_gse']:
        for n in G.nodes:
            if att in G.nodes[n]:
                G.nodes[n][att] = float(G.nodes[n][att])

    # 2. Crear un diccionario de atributos de nodos
    node_attrs_dict = {}

    for node_id in G.nodes:
        node_attrs = G.nodes[node_id]
        node_attrs_dict[node_id] = {**node_attrs}

    # 3. Crear listas de nodos y aristas
    nodes = []
    edges = []
    edge_attributes = []

    for source, target, edge_attrs in G.edges(data=True):
        edges.append([source, target])
        edge_attributes.append([float(edge_attrs['length']), float(edge_attrs['travel_time'])])

    for node_id in G.nodes: 
        node_attrs = node_attrs_dict[node_id]
        nodes.append([node_attrs['y'], node_attrs['x'], node_attrs['beautiful'], node_attrs['boring'], node_attrs['depressing'],
                    node_attrs['lively'], node_attrs['safe'], node_attrs['wealthy'], node_attrs['bc'], node_attrs['eigen'], node_attrs['pg_rank'], node_attrs['division'], node_attrs['ismt'],
                    node_attrs['houses'], node_attrs['hog_40pct'], node_attrs['pct_hog40p'], node_attrs['ave_gse']])
    
    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of edges: {len(edges)}")
    print(f"Number of edge attributes: {len(edge_attributes)}")
    
    # 4. Crear objeto Data de PyTorch Geometric
    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attributes = torch.tensor(edge_attributes, dtype=torch.float)

    print(f"Size of x tensor: {x.size()}")
    print(f"Size of edge_index tensor: {edge_index.size()}")
    print(f"Size of edge_attributes tensor: {edge_attributes.size()}")
    
    # Crear el objeto Data para el grafo
    data = Data(x=x,
                edge_index=edge_index,
                edge_attributes=edge_attributes)
    
    torch.save(data, f'Data/{file_name}.pt')
    print(f"Graph saved as PyTorch in Data/{file_name}.pt")
    

def random_filling(X):
    return torch.randn_like(X)


def zero_filling(X):
    return torch.zeros_like(X)


def mean_filling(X, feature_mask):
    n_nodes = X.shape[0]
    return compute_mean(X, feature_mask).repeat(n_nodes, 1)


def neighborhood_mean_filling(edge_index, X, feature_mask):
    n_nodes = X.shape[0]
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    edge_values = torch.ones(edge_index.shape[1]).to(edge_index.device)
    edge_index_mm = torch.stack([edge_index[1], edge_index[0]]).to(edge_index.device)

    D = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, feature_mask.float())
    mean_neighborhood_features = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, X_zero_filled) / D
    # If a feature is not present on any neighbor, set it to 0
    mean_neighborhood_features[mean_neighborhood_features.isnan()] = 0

    return mean_neighborhood_features


def feature_propagation(edge_index, X, feature_mask, num_iterations):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)

    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask)


def filling(filling_method, edge_index, X, feature_mask, num_iterations=None):
    if filling_method == "random":
        X_reconstructed = random_filling(X)
    elif filling_method == "zero":
        X_reconstructed = zero_filling(X)
    elif filling_method == "mean":
        X_reconstructed = mean_filling(X, feature_mask)
    elif filling_method == "neighborhood_mean":
        X_reconstructed = neighborhood_mean_filling(edge_index, X, feature_mask)
    elif filling_method == "feature_propagation":
        X_reconstructed = feature_propagation(edge_index, X, feature_mask, num_iterations)
    else:
        raise ValueError(f"{filling_method} method not implemented")
    return X_reconstructed


def get_feature_propagation(data):
    n_nodes, n_features = data.x.shape
    num_iterations = 40

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    missing_feature_mask = torch.where(data.x == 0, 0.0, 1.0).bool().to(device)
    x = data.x.clone()
    x[~missing_feature_mask] = float("nan")

    print("Starting feature filling")
    start = time.time()
    filled_features = (
        filling("feature_propagation", data.edge_index, x, missing_feature_mask, num_iterations,)
        if "gcn" not in ["gcnmf", "pagnn"]
        else torch.full_like(x, float("nan"))
    )
    print(filled_features)
    print(f"Feature filling completed. It took: {time.time() - start:.2f}s")
    return torch.where(missing_feature_mask, data.x, filled_features)

def compute_mean(X, feature_mask):
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    num_of_non_zero = torch.count_nonzero(feature_mask, dim=0)
    mean_features = torch.sum(X_zero_filled, axis=0) / num_of_non_zero
    # If a feature is not present on any node, set it to 0
    mean_features[mean_features.isnan()] = 0

    return mean_features


### CLUSTERING ####

def kmeans_clustering(latent_representation, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=27)
    cluster_labels = kmeans.fit_predict(latent_representation)

    return cluster_labels

def plot_clusters(latitudes, longitudes, cluster_labels, node_size=4, c_map='viridis'):
    # Crear un diccionario para almacenar los nodos de cada cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    cmap = plt.get_cmap(c_map)

    plt.figure(figsize=(6, 6))
    for cluster_label, nodes in clusters.items():
        cluster_latitudes = [latitudes[i] for i in nodes]
        cluster_longitudes = [longitudes[i] for i in nodes]
        plt.scatter(cluster_longitudes, cluster_latitudes, s=node_size, label=f'Cluster {cluster_label}', c=cmap(cluster_label / len(clusters)))

    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Clusters Santiago')
    plt.legend()
    plt.show()