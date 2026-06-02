# Network Traffic Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Paper](https://img.shields.io/badge/Paper-Connection%20Science-green.svg)](https://doi.org/10.1080/09540091.2020.1870437)
[![GitHub stars](https://img.shields.io/github/stars/pritom007/Network-Traffic-Classification?style=social)](https://github.com/pritom007/Network-Traffic-Classification/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/pritom007/Network-Traffic-Classification?style=social)](https://github.com/pritom007/Network-Traffic-Classification/forks)
[![DOI](https://img.shields.io/badge/DOI-10.1080%2F09540091.2020.1870437-blue)](https://doi.org/10.1080/09540091.2020.1870437)

Machine-learning experiments for classifying network traffic flows collected from a Docker-based SDN lab network. The dataset contains more than 300,000 flows analyzed with nDPI, grouped from 100+ application protocols into 10 traffic classes.

This repository includes the original research artifacts, saved models, confusion matrices, and a cleaner Python workflow for reproducing common experiments.

## Project Impact

As of June 2, 2026, this public research repository has accumulated measurable academic and open-source interest:

| Signal | Count | Notes |
| --- | ---: | --- |
| GitHub stars | 40 | First public star recorded on March 10, 2021. |
| GitHub forks | 11 | Fork activity spans September 2020 through September 2025. |
| Crossref cited-by count | 14 | Citation metadata for the associated Connection Science article. |
| Google Scholar profile | Available | Author profile provides a complementary citation view. |
| Paper references | 42 | References registered in Crossref metadata. |

See [Repository Impact](docs/REPOSITORY_IMPACT.md) for the star/fork timeline, citation graph, and data-source notes.

## Research Summary

The project evaluates supervised machine-learning models for flow-level network traffic classification using seven selected flow features:

| Feature group | Columns |
| --- | --- |
| Protocol metadata | `protocol`, `src_port`, `dst_port` |
| Packet counts | `src2dst_packets`, `dst2src_packets` |
| Byte counts | `src2dst_bytes`, `dst2src_bytes` |

Reported results from the original experiments:

| Method | Accuracy |
| --- | ---: |
| Decision Tree | 95.80% |
| Random Forest | 96.69% |
| KNN | 97.24% |
| PAA | 99.29% |

For the full methodology, class grouping, and experimental setup, read the paper:

> P. K. Mondal, L. P. Aguirre Sanchez, E. Benedetto, Y. Shen, and M. Guo, "A dynamic network traffic classifier using supervised ML for a Docker-based SDN network," Connection Science, 2021. https://doi.org/10.1080/09540091.2020.1870437

## Repository Layout

```text
.
├── DecisionTree/                  # Original decision-tree scripts, notebook, model, outputs
├── RandomForest/                  # Original random-forest scripts, model, outputs
├── KNN/                           # Original KNN scripts, model, outputs
├── DNN/                           # Original neural-network scripts, model, outputs
├── Dataset/                       # Dataset access notes
├── network_traffic_classification/ # Modern reusable Python package
├── docs/                          # Project documentation
├── dictionary.py                  # Legacy protocol-to-class helper
├── test.txt                       # Protocol-to-class mapping used by legacy scripts
└── README.md
```

## Quick Start

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the legacy DNN scripts, install the optional TensorFlow dependency:

```bash
pip install -r requirements-dnn.txt
```

Train a model with the modern CLI:

```bash
python -m network_traffic_classification train \
  --data path/to/total_class.csv \
  --model random-forest \
  --output models/random_forest.joblib
```

Available model names:

```text
decision-tree
random-forest
knn
```

The CLI prints accuracy and a classification report. It can also save the trained model and class labels:

```bash
python -m network_traffic_classification train \
  --data path/to/total_class.csv \
  --model knn \
  --output artifacts/knn.joblib \
  --labels-output artifacts/classes.txt \
  --test-size 0.33 \
  --random-state 42
```

## Dataset

The raw `.pcap` files and processed CSV are large, so the dataset is not committed to this repository. See [Dataset/How to get the data.txt](Dataset/How%20to%20get%20the%20data.txt) for access instructions and research-use conditions.

Expected labeled training file:

```text
total_class.csv
```

Expected columns:

```text
#flow_id, protocol, src_ip, src_port, dst_ip, dst_port, ndpi_proto_num,
src2dst_packets, src2dst_bytes, dst2src_packets, dst2src_bytes,
ndpi_proto, class
```

## Original Scripts

The original scripts are preserved for traceability:

```bash
python DecisionTree/decisiontree.py
python RandomForest/randomforest.py
python KNN/knn.py
python DNN/dnn.py
```

Some legacy scripts contain machine-specific Windows paths. Prefer the modern CLI for new experiments, or update `file_dir` in the legacy scripts before running them.

## Citation

If this repository or dataset supports your work, please cite:

```bibtex
@article{mondal2021dynamic,
  title={A dynamic network traffic classifier using supervised ML for a Docker-based SDN network},
  author={Mondal, Pritom Kumar and Aguirre Sanchez, Lizeth P. and Benedetto, Emmanuele and Shen, Yao and Guo, Minyi},
  journal={Connection Science},
  pages={1--26},
  year={2021},
  publisher={Taylor \& Francis},
  doi={10.1080/09540091.2020.1870437}
}
```

## Contributing

Contributions are welcome, especially improvements to reproducibility, documentation, and model evaluation. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
