# dimon: Your Canonical Answer Management Tool
dimon is a command-line interface (CLI) tool for evaluating and managing golden dataset for evaluation your retrival mechnism.

Installation
You can install dimon directly from PyPI:

```
pip install dimon
```



## The CLI tool provides three main commands:

### evaluate
This command evaluates a specified model on a given dataset. It computes embeddings for each document in the corpus, retrieves the top-k most similar documents for each query, and calculates the Mean Reciprocal Rank (MRR) for evaluation.

```
dimon evaluate --database_url 'http://localhost:9200' --dataset 'golden_dataset.csv'

```

### diff
This command reads a report file, filters out entries with MRR below a certain threshold

```
dimon diff --file_name 'mrr_report.csv' --threshold 0.2

```

### merge
This command reads a report file, filters out entries marked as 'Accept', and simulates a merging process.

```

dimon merge --report_name 'to_confirmed_change.csv'


```
