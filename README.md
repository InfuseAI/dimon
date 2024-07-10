# dimon: Your Canonical Answer Management Tool
dimon is a command-line interface (CLI) tool for evaluating and managing golden dataset for evaluation your retrival mechnism.

## Installation
You can install dimon directly from PyPI:

```
pip install dimon
```

## Prepare env file

Create a file named `.env`, or copy it from `.env.example`.
This file should contain the default configuration values for dimon. Here is an example of what the `.env` file might look like:

```
DATABASE_URL=http://localhost:9200
DATABASE_INDEX=your_index
USERNAME=elastic
PASSWORD=your_password_here
DATASET_PATH=/path/to/your/dataset
```

Replace the values with the actual ones.

Note: Currently, the script only supports ElasticSearch as the vector database. Ensure that your ElasticSearch instance is correctly set up and accessible via the URL provided in the DATABASE_URL parameter. The DATABASE_INDEX should point to the specific index within ElasticSearch that you want to use for storing and querying vectors.


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
