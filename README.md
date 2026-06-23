# Protein Melting Point Prediction with Graph Neural Networks

A graph neural network project for predicting protein melting temperatures from AlphaFold-predicted protein structures.

This project converts protein structures into residue-level graphs, trains a PyTorch Geometric graph neural network to predict melting temperature, and generates visualizations showing both model performance and learned structural importance.

## Overview

Protein stability is a central problem in biochemistry, protein engineering, and drug discovery.

A model that can estimate melting temperature from structure could help with:

- screening thermostable proteins
- comparing homologous proteins across species
- understanding structure-stability relationships
- prioritizing proteins for experimental validation
- exploring how geometry affects thermal stability

This project is a small-scale prototype of that idea using graph neural networks.

Protein melting temperature, often written as `Tm`, is the temperature at which a protein begins to lose its folded structure. Higher melting temperatures generally indicate greater thermal stability.

This project explores whether 3D structural information from AlphaFold models can be used to predict protein melting temperatures across species.

The high-level view of the pipeline is:

1. Load protein melting temperature data from `cross-species.csv`
2. Match protein IDs to local AlphaFold structure files
3. Convert each protein structure into a graph
4. Train a graph neural network regression model
5. Evaluate prediction accuracy
6. Visualize learned structural importance using saliency maps

## How the model works

### 1. Protein structures become graphs

Each protein is represented as a graph:

- **Nodes**: amino acid residues
- **Node features**: one-hot encoded amino acid identities
- **Edges**: spatial proximity between residues
- **Positions**: 3D coordinates from AlphaFold structures

The script uses the alpha carbon coordinate of each residue as its 3D position. Two residues are connected by an edge if they are within a chosen distance threshold.

By default, the graph construction uses an 8 Å distance cutoff.

### 2. The GNN predicts melting temperature

The model is a graph convolutional neural network built with PyTorch Geometric.

The architecture uses:

- three `GCNConv` layers
- ReLU activations
- global mean pooling over graph nodes
- dropout regularization
- a final linear layer for regression

The model outputs a single predicted melting temperature in degrees Celsius.

### 3. Training and evaluation

The dataset is split into training and validation sets. The model is trained using mean squared error loss, and performance is reported as root mean squared error, or RMSE, in degrees Celsius.

The script also generates:

- a training curve
- an actual-vs-predicted scatter plot
- a 3D protein graph visualization
- an interactive saliency heatmap

## Example outputs

### Learning curve

`learning_curve.png` shows how training and validation RMSE change over the training process.

### Actual vs. predicted melting temperatures

`actual_vs_predicted.png` compares experimentally measured melting temperatures against model predictions.

Points closer to the diagonal line represent better predictions.

### Protein graph visualization

`protein_graph.png` shows an example protein represented as a 3D graph.

### Interactive importance heatmap

`interactive_heatmap.html` visualizes which residues the trained model treats as most important for a prediction.

This is generated using a saliency-style gradient analysis.

## Installation

Clone the repository:

```bash
git clone https://github.com/anisrikanth/Protein-Melting-Points.git
cd Protein-Melting-Points
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install pandas numpy matplotlib networkx plotly biopython torch
```

PyTorch Geometric may require a separate install command depending on your operating system, Python version, CUDA version, and PyTorch version.

For CPU-only usage, the install usually looks like:

```bash
pip install torch-geometric
```

If that fails, follow the official PyTorch Geometric installation instructions for your machine.

## Running the project

Make sure the following are present in the project root:

```txt
cross-species.csv
alphafold_data/
```

Then run:

```bash
python alpha_gnn.py
```

The script will:

1. scan the local AlphaFold structure folder
2. match protein IDs against the melting temperature CSV
3. build PyTorch Geometric protein graphs
4. train the GNN
5. save visualizations to disk

## Data

The project expects a CSV containing protein IDs and melting temperatures.

The script looks for columns such as:

- `Protein_ID`
- `target_id`
- `uniprot_id`
- `meltPoint`
- `Tm`
- `melting_temp`

Protein IDs are matched against AlphaFold structure filenames in the `alphafold_data/` directory.

The expected AlphaFold filename format is similar to:

```txt
AF-<UNIPROT_ID>-F1-model_v6.pdb.gz
```

## Model details

The main model is defined as:

```python
class ProteinRegressorGNN(torch.nn.Module):
    ...
```

It uses:

```python
GCNConv -> ReLU
GCNConv -> ReLU
GCNConv
global_mean_pool
dropout
linear regression head
```

The final output is a single continuous value: predicted protein melting temperature.

## Limitations

This is a small experimental prototype I did during my time at Recurse Center, not a research-grade model.

Current limitations include:

- The model uses AlphaFold-predicted structures, not necessarily experimentally solved structures.
- Residue features are currently simple one-hot amino acid encodings.
- Edges are based only on spatial distance.
- The model does not explicitly include solvent accessibility, secondary structure, pLDDT confidence, electrostatics, sequence conservation, or organism-level features.
- The train/validation split is random, so homologous proteins may appear across splits.
- The dataset size depends on which proteins have both melting temperature labels and matching local AlphaFold files.
- RMSE should be interpreted carefully because protein melting temperature prediction is noisy and experimentally variable.


## Required Packages

- PyTorch
- PyTorch Geometric
- Biopython
- pandas
- NumPy
- Matplotlib
- NetworkX
- Plotly
