# CoMGL: Collaborative Multi-view Graph Learning for Strict Cold-start Tasks

- Main view: ("paper", "to", "author")
- Auxiliary views: ("paper", "to", "Conference"), ("paper", "to", "term")

## Data 

### Data Preparation
The dataset is placed in the `./data/` directory according to the `OGB Dataset` format.

Template class for the `dataset`
```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Create node types "paper" and "author" holding a feature matrix:
data['paper'].x = torch.randn(num_papers, num_paper_features)
data['author'].x = torch.randn(num_authors, num_authors_features)

# Create an edge type "(author, writes, paper)" and building the
# graph connectivity:
data[("paper", "to", "author")].edge_index = ...  # [2, num_edges]

```

### Data Process

- To satisfy the inductive setting, divide the train/val/test datasets, remove the redundant edges in the Main view of each dataset, but keep all the edges in the Auxiliary view;
- Remove all edges in the Main view of the val/test datasets to satisfy the "Strict Cold Start" scenario;

![](https://cdn.jsdelivr.net/gh/Zaiyun27/Imgur/img/202211132039830.png)

## Model

- Heterogeneous Grpah
    - Encoder
        - Graph SAGE: Each view establishes a separate heterogeneous graph and corresponding Encoder
    - Aggregator
        - Uncertainty Estimate / Attention: Aggregate Auxiliary view's paper representation for the prediction task of the Main view
    - Predictor
        - Main view Predictor: Link prediction / Node classification
        - Auxiliary view Predictor: Link prediction for auxiliary tasks


## Experiments

### 1. Node Prediction

1. Auxiliary views encode
2. Auxiliary views’ paper embedding aggregate
3. Main view edge generation
4. Main view encode
5. Main view’s node prediction task

```bash
python main.py --exp_mode node_prediction --data_path ./data
```

### 2. Link Prediction

1. Auxiliary views encode
2. Auxiliary views’ paper embedding aggregate
3. Main view’s link prediction task

```bash
python main.py --exp_mode link_prediction --data_path ./data
```

