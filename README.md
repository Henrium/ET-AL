# Entropy-Targeted Active Learning

This repository contains an implementation of entropy-targeted active learning (ET-AL) for materials data bias mitigation, associated with our paper.

## Copyright
This code is open-sourced under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Zhang, H., Chen, W. W., Rondinelli, J. M., and Chen W. (2022). ET-AL: entropy-targeted active learning for bias mitigation in materials data. arXiv preprint arXiv:2211.07881.

```
@misc{zhang2022etal,
    author = {Zhang, Hengrui and Chen, Wei Wayne and Rondinelli, James M. and Chen, Wei},
    title = {ET-AL: entropy-targeted active learning for bias mitigation in materials data},
    howpublished = {arXiv preprint arXiv:2204.10532},
    year = {2022},
    doi = {10.48550/ARXIV.2211.07881},
    url = {https://arxiv.org/abs/2211.07881},
    publisher = {arXiv}
}
```

## Descriptions
`etal_gp_jarvis.py` implements the ET-AL algorithm and demonstrates on the Jarvis-CFID dataset.

`ML_comparison` compares several ML models on different training sets.

`datasets/` provides data required for reproducing the results in our paper.

`notebooks/` contains Jupyter Notebooks for pre- and post-processing:

- `Jarvis_data` is used for retrieving, cleaning the Jarvis CFID data and generating graph embeddings.
- `Jarvis_featurize` generates physical descriptors for the Jarvis CFID data.
- `plot_data` is used for creating relevant plots for visualization.`results/` contains data generated in ET-AL demonstration on the Jarvis-CFID dataset

`utils/` contains code for generating representations of materials structures:

- `compound_featurizer.py` for physical descriptors
- `cgcnn/` for graph embeddings

## Usage
### Requirements

### Data preparation
Organize the dataset in a Data Frame and change the data paths in `etal_main.py`. For demonstration purposes, a dataset derived from the Jarvis CFID data is provided in `datasets/`: the crystal structures and properties are in `data_cleaned.pkl`, and the graph embeddings are in `cgcnn_embeddings.pkl`.

### Run ET-AL model



