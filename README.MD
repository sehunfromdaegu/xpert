# xPerT: Extended Persistence Transformer

This repository contains code for training a GNN model (`GIN`) using `PyTorch Geometric` and generating extended persistence diagrams from graph datasets with `giotto-deep`.

## Installation

Follow the steps below to set up the required environment:

```bash
# Create a new conda environment
conda create --name xpert python=3.9
conda activate xpert
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
python -m pip install -U giotto-deep
conda install pyg=2.5.2 -c pyg
```
## Optional: Fix scikit-learn errors

If you encounter any errors related to scikit-learn, you can reinstall the correct version as follows:

```bash
pip uninstall scikit-learn
pip install scikit-learn==1.1.1
```

## Library modification
There are issues with specific versions of PyTorch Geometric and giotto-deep. To resolve these, apply the following fixes:

Locate the library path by running:
```bash
pip show torch_geometric
```
The library path will be listed under Location: PATH_TO_LIBS.


#### PyTorch Geometric
Open the file /PATH_TO_LIBS/torch_geometric/io/fs.py and modify line 193 as follows:
```bash
fs1.mv(path1, path2, recursive)  # Original
fs1.mv(path1, path2)             # Updated
```

#### giotto-deep
- Open the file /PATH_TO_LIBS/gdeep/data/datasets/persistence_diagrams_from_builder.py
- Modify line 206 to correct the graph labeling logic:
```bash
labels = (np.loadtxt(graph_labels, delimiter=",", dtype=np.int32).T + 1) // 2  # Original
labels = np.loadtxt(graph_labels, delimiter=",", dtype=np.int32).T             # Updated
```

## Classification on Graph Datasets
To perform classification on graph datasets, use the following command:

```bash
python graph_classification.py --dataname <dataset> --model <modelname>
```

- `dataset`: Choose from the available options:
  - 'IMDB-BINARY'
  - 'IMDB-MULTI'
  - 'MUTAG'
  - 'PROTEINS'
  - 'COX2'
  - 'DHFR'
  
- `modelname`: Specify the model to be used:
  - 'xpert' (Extended Persistence Transformer)
  - 'gin' (Graph Isomorphism Network)
  - 'gin_assisted_concat' (GIN + xPerT by concat representations)
  - 'gin_assisted_sum' (GIN + xPerT by summing representations)

For example, to train the xPerT model on the MUTAG dataset, run:

```bash
python graph_classification.py --dataname MUTAG --model xpert
```

## Classification on ORBIT dataset
To perform classification on ORBIT5K datasets, use the following command:

```bash
python orbit_classification.py
```

To perform classification on ORBIT5K datasets, use the following command:

```bash
python orbit_classification.py --samples_per_class 20000
```



