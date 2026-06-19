import os
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.utils.data as data_utils

import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.utils import to_networkx

import plotly.graph_objects as go

AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def pdb_to_graph(pdb_filepath, distance_threshold=8.0):
    """Reads a compressed PDB file directly from the hard drive and builds a graph."""
    parser = PDBParser(QUIET=True)
    try:
        with gzip.open(pdb_filepath, 'rt') as f:
            structure = parser.get_structure('protein', f)
    except Exception as e:
        print(f"  [Parse Error] {pdb_filepath}: {e}")
        return None
        
    coords, node_features = [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.resname in AA_TO_INT and 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
                    feature_vec = np.zeros(len(AMINO_ACIDS))
                    feature_vec[AA_TO_INT[residue.resname]] = 1.0
                    node_features.append(feature_vec)

    if len(coords) < 10: 
        print(f"  [Too Short] {pdb_filepath} only had {len(coords)} CA atoms.")
        return None

    coords = np.array(coords)
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    sources, targets = np.where((dist_matrix < distance_threshold) & (dist_matrix > 0))
    
    edge_index = torch.tensor(np.array([sources, targets]), dtype=torch.long)
    
    pos = torch.tensor(coords, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, pos=pos)

def visualize_protein_3d(data, filename="my_cool_protein_graph.png"):
    """Generates a beautiful neon 3D scatter plot of the protein graph."""
    print(f"\n[VISUALS] Rendering 3D protein graph to {filename}...")
    G = to_networkx(data, to_undirected=True)
    pos = data.pos.numpy()

    fig = plt.figure(figsize=(10, 10), facecolor='#0f172a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f172a')

    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    for edge in G.edges():
        i, j = edge
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], color='#00ffcc', alpha=0.3, linewidth=1.5)

    ax.scatter(xs, ys, zs, c='#ff00ff', s=50, edgecolors='white', linewidths=0.5, alpha=0.9)

    ax.set_axis_off()
    plt.title("Protein as a 3D Graph (Nodes=Atoms, Edges=Proximity)", color='white', fontsize=16, y=1.05)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

def plot_training_curve(train_history, val_history, filename="ai_learning_curve.png"):
    """Plots the learning progression of the AI."""
    print(f"[VISUALS] Rendering training curve to {filename}...")
    plt.figure(figsize=(10, 6), facecolor='white')
    plt.plot(train_history, label="Train Error (RMSE)", color="#ff00ff", linewidth=3)
    plt.plot(val_history, label="Validation Error (RMSE)", color="#00ffcc", linewidth=3)
    plt.xlabel("Training Epochs", fontsize=12, fontweight='bold')
    plt.ylabel("Error in Celsius (°C)", fontsize=12, fontweight='bold')
    plt.title("GNN Thermostability Learning Progression", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_actual_vs_predicted(actuals, predicteds, filename="actual_vs_predicted.png"):
    """Feature A: The classic Machine Learning Accuracy Scatter Plot."""
    print(f"[VISUALS] Rendering Actual vs Predicted scatter plot to {filename}...")
    plt.figure(figsize=(8, 8), facecolor='white')
    
    plt.scatter(actuals, predicteds, alpha=0.6, color="#ff00ff", edgecolors="#0f172a")

    min_val = min(min(actuals), min(predicteds))
    max_val = max(max(actuals), max(predicteds))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction")

    plt.xlabel("Actual Laboratory Melting Temp (°C)", fontsize=12, fontweight='bold')
    plt.ylabel("AI Predicted Melting Temp (°C)", fontsize=12, fontweight='bold')
    plt.title("GNN Accuracy: Actual vs Predicted", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def get_saliency_map(model, data, device):
    """Feature C (Math): Calculates which atoms the AI thinks are most important."""
    model.eval()
    
    data = data.to(device)
    data.x.requires_grad_()
    
    batch = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
    
    out = model(data.x, data.edge_index, batch).squeeze()
    
    out.backward()
    
    saliency = data.x.grad.abs().sum(dim=1).cpu().numpy()
    
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

def visualize_interactive_heatmap(data, saliency_scores, filename="interactive_heatmap.html"):
    """Features B & C: Renders a spinning HTML 3D graph colored by AI Attention."""
    print(f"[VISUALS] Rendering Interactive 3D Attention Heatmap to {filename}...")
    pos = data.pos.numpy()
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    edge_x, edge_y, edge_z = [], [], []
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        edge_x.extend([xs[src], xs[dst], None])
        edge_y.extend([ys[src], ys[dst], None])
        edge_z.extend([zs[src], zs[dst], None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.15)', width=2),
        hoverinfo='none'
    )

    node_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(
            size=6,
            color=saliency_scores, # This maps our PyTorch math to colors!
            colorscale='Turbo',    # Blue (Cold/Unimportant) to Red (Hot/Important)
            colorbar=dict(
                title=dict(text="AI Importance Score", font=dict(color="white")), 
                tickfont=dict(color="white")
            ),
            opacity=0.9
        ),
        text=[f"Atom {i} | AI Importance: {s:.2f}" for i, s in enumerate(saliency_scores)],
        hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(text="What the AI Sees: Geometric Importance Map", font=dict(color='white', size=20)),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.write_html(filename)


class ProteinRegressorGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(ProteinRegressorGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, pdb_dir, uid_to_file):
        self.pdb_dir = pdb_dir
        self.graphs = []
        self.labels = []
        
        print(f"Building PyTorch Graphs for {len(dataframe)} proteins...")
        for idx, (i, row) in enumerate(dataframe.iterrows()):
            uid = row['uniprot_id']
            temp = row['melting_temp']
            
            exact_filename = uid_to_file.get(uid)
            if not exact_filename:
                continue
                
            pdb_path = os.path.join(self.pdb_dir, exact_filename)
            
            graph = pdb_to_graph(pdb_path)
            if graph:
                self.graphs.append(graph)
                self.labels.append(temp)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx+1}/{len(dataframe)} structures...")
                
        print(f"Successfully loaded {len(self.graphs)} valid graphs.")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = torch.tensor([self.labels[idx]], dtype=torch.float)
        return graph, label



def main():
    csv_path = "cross-species.csv"
    pdb_dir = "alphafold_data"
    
    if not os.path.exists(csv_path) or not os.path.exists(pdb_dir):
        print("Please ensure 'cross-species.csv' and the 'alphafold_data' folder are here.")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    id_col = next((c for c in ['Protein_ID', 'target_id', 'uniprot_id'] if c in df.columns), df.columns[1])
    tm_col = next((c for c in ['meltPoint', 'Tm', 'melting_temp'] if c in df.columns), df.columns[3])
    
    df = df.rename(columns={id_col: 'uniprot_id', tm_col: 'melting_temp'})
    df['uniprot_id'] = df['uniprot_id'].astype(str).str.split('_').str[0].str.strip().str.upper()
    df['melting_temp'] = pd.to_numeric(df['melting_temp'], errors='coerce')
    df = df.dropna(subset=['uniprot_id', 'melting_temp'])

    print("2. Scanning local AlphaFold files...")
    available_files = os.listdir(pdb_dir)
    
    uid_to_file = {}
    for f in available_files:
        if f.startswith("AF-") and ".pdb.gz" in f:
            uid = f.split("-")[1].upper()
            uid_to_file[uid] = f
            
    available_ids = set(uid_to_file.keys())
    
    matched_df = df[df['uniprot_id'].isin(available_ids)]
    
    matched_df = matched_df.groupby('uniprot_id', as_index=False)['melting_temp'].mean()
    print(f"3. Matched {len(matched_df)} valid proteins between CSV and local folder!")
    
    if len(matched_df) < 10:
        print("Not enough matches to train. Ensure you downloaded the right .tar archive.")
        return

    sample_size = min(3000, len(matched_df))
    matched_df = matched_df.sample(n=sample_size, random_state=42)

    dataset = ProteinDataset(matched_df, pdb_dir, uid_to_file)
    
    if len(dataset) < 2:
        print("\nFATAL ERROR: 0 graphs loaded. Check the [Parse Error] logs above.")
        return
        
    visualize_protein_3d(dataset[0][0])
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = data_utils.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProteinRegressorGNN(num_node_features=20, hidden_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.MSELoss()
    
    epochs = 100
    
    train_history = []
    val_history = []
    
    final_actuals = []
    final_preds = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_graphs, batch_labels in train_loader:
            batch_graphs, batch_labels = batch_graphs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            out = model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch).squeeze()
            batch_labels = batch_labels.squeeze()
            if out.dim() == 0: out, batch_labels = out.unsqueeze(0), batch_labels.unsqueeze(0)
            
            loss = criterion(out, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_graphs.num_graphs
            
        avg_train_rmse = np.sqrt(train_loss / max(train_size, 1))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_graphs, batch_labels in val_loader:
                batch_graphs, batch_labels = batch_graphs.to(device), batch_labels.to(device)
                out = model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch).squeeze()
                batch_labels = batch_labels.squeeze()
                if out.dim() == 0: out, batch_labels = out.unsqueeze(0), batch_labels.unsqueeze(0)
                
                loss = criterion(out, batch_labels)
                val_loss += loss.item() * batch_graphs.num_graphs
                
                if epoch == epochs - 1:
                    final_preds.extend(out.cpu().numpy())
                    final_actuals.extend(batch_labels.cpu().numpy())
                
        avg_val_rmse = np.sqrt(val_loss / max(val_size, 1))
        
        train_history.append(avg_train_rmse)
        val_history.append(avg_val_rmse)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train RMSE: ±{avg_train_rmse:.2f}°C | Val RMSE: ±{avg_val_rmse:.2f}°C")
    
    plot_training_curve(train_history, val_history)
    
    plot_actual_vs_predicted(final_actuals, final_preds)
    
    sample_graph, _ = dataset[0]
    sample_graph = sample_graph.clone() # Clone it so we don't mess up the original
    
    saliency_scores = get_saliency_map(model, sample_graph, device)
    visualize_interactive_heatmap(sample_graph, saliency_scores)
    
if __name__ == "__main__":
    main()