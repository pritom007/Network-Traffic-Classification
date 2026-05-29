# Project Structure

This repository keeps the original research files while adding a small reusable Python package for repeatable experiments.

## Original Research Artifacts

| Path | Purpose |
| --- | --- |
| `DecisionTree/` | Decision-tree training script, prediction script, notebook, saved model, and confusion matrices. |
| `RandomForest/` | Random-forest training script, prediction script, saved model, and confusion matrix. |
| `KNN/` | KNN training script, prediction script, saved model, and confusion matrix. |
| `DNN/` | Neural-network training and prediction scripts, saved models, and confusion matrix. |
| `Dataset/` | Dataset access notes and research-use conditions. |
| `test.txt` | Mapping from nDPI protocol names to grouped traffic classes. |

## Modern Package

| Path | Purpose |
| --- | --- |
| `network_traffic_classification/config.py` | Shared column names and feature definitions. |
| `network_traffic_classification/data.py` | Dataset loading and validation helpers. |
| `network_traffic_classification/models.py` | Model factory for supported scikit-learn classifiers. |
| `network_traffic_classification/cli.py` | Command-line interface for training and evaluation. |

## Recommended Workflow

1. Obtain the dataset using the instructions in `Dataset/`.
2. Create a virtual environment and install `requirements.txt`.
3. Run experiments through `python -m network_traffic_classification train`.
4. Save generated models and reports under an ignored local folder such as `artifacts/`.
5. Report metrics with the command, random seed, and dataset version used.
