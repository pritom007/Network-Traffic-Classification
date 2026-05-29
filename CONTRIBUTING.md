# Contributing

Thank you for helping improve this research repository.

## Good First Contributions

- Improve setup and reproducibility instructions.
- Add evaluation results with the same train/test protocol.
- Clean legacy scripts without changing their original intent.
- Add notebooks that clearly state dataset version, features, model parameters, and metrics.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the modern training CLI:

```bash
python -m network_traffic_classification train --data path/to/total_class.csv --model random-forest
```

## Pull Request Checklist

- Keep large datasets, generated reports, and local model outputs out of Git.
- Document the dataset split, model parameters, and random seed used for new results.
- Include the exact command used to reproduce any reported metric.
- Preserve citation and dataset usage requirements.
