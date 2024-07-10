import numpy as np
import click
import time
import re
import os
import json
import pandas as pd
from dataclasses import dataclass

from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress import Progress


@dataclass
class Entry:
    query_id: int
    question: str
    expected_id: str
    expected: str
    actual_id: str
    actual: str
    mrr: float

@click.group()
def cli():
    pass

load_dotenv()

@cli.command()
@click.option('--database_url', '-db', default=os.getenv('DATABASE_URL', 'h'), help='Database URL')
@click.option('--database_index', '-index', default=os.getenv('DATABASE_INDEX', ''), help='Database index')
@click.option('--username', '-u', default=os.getenv('USERNAME', ''), help='Username for the database')
@click.option('--password', '-p', default=os.getenv('PASSWORD', ''), help='Password for the database')
@click.option('--dataset', '-d', default=os.getenv('DATASET_PATH', ''), type=click.Path(exists=True), help='Path to the dataset')
def evaluate(dataset, database_url, database_index, username, password, top_k=10, verbose=False):
    d = pd.read_csv(dataset)

    # Prepare to collect evaluation results
    eval_results = []
    console = Console()
    # nomic embedding model
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # ollama
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)
    vector_store = ElasticsearchStore(
            index_name=database_index,
            es_url=database_url,
            es_user=username,
            es_password=password,
        )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    # count
    counts = {}

    with Live(console=console, screen=True, auto_refresh=False) as live:
        counter = 0  # Add a counter
        display_interval = console.size.height - 4  # Number of queries to process before updating the display
        processes = []  
        # Iterate over each query in the dataset to evaluate retrieval performance
        for index, row in d.iterrows():
            try:
                query_id = row['query_id']
                query = row['query']
                expected_id = row['expected_id']
                expected_text = row['expected_text']
            except ValueError:
                continue  # Skip this iteration and move on to the next one

            # Retrieve the top_k most similar documents for the current query and extract the IDs of the retrieved documents
            nodes = retriever.retrieve(query) 
            retrieved_ids = [result.node_id for result in nodes]
            retrieved_text = re.sub(r'\W+', ' ', nodes[0].text).strip()
            
            is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

            # Calculate the Mean Reciprocal Rank (RR) and append to results
            if is_hit:
                rank = retrieved_ids.index(expected_id) + 1
                mrr = 1 / rank

                for r in retrieved_ids[:3]:
                    if not r in counts:
                        counts[r] = 0
                    counts[r] = counts[r] + 1
            else:
                mrr = 0
            eval_results.append(mrr)
            
            #Generate a process for the current query and add it to the list of processes
            process = _generate_entry(query_id, query, expected_id, expected_text, retrieved_ids[0], retrieved_text, mrr)
            processes.append(process)
                
            counter += 1  # Increment the counter
            if counter % display_interval == 0:  # Only update the display every display_interval queries
                 tmp = sorted(processes[counter - display_interval : counter], key=lambda p: p.mrr, reverse=True)
                 live.update(_create_process_table(console, tmp), refresh=True)

        if eval_results:
            final_table = Table(style="bold")
            final_table.add_row("AVG MRR", "", "", "", f"{np.average(eval_results):.1f}")
            live.update(final_table)
        time.sleep(1)

    data = [vars(p) for p in processes]
    df = pd.DataFrame(data)
    # TODO: specify output file name
    df.to_csv("mrr_report.csv", index=False)
    vector_store.close()

    with open('count.json', 'w') as f:
        json.dump(counts, f)

    # Return the average MRR across all queries as the final evaluation metric
    return np.average(eval_results)


@cli.command()
@click.option('--file_name', '-f', default='mrr_report.csv', required=True)
@click.option('--threshold', '-t', default=0.7, required=True)
def diff(file_name, threshold):
    df = pd.read_csv(file_name)
    df = df[df['mrr'] < threshold ]
    df['Accept / Reject'] = ''
    df.to_csv('to_confirmed_change.csv')

@cli.command()
@click.option('--report', '-r', default='to_confirmed_change.csv')
@click.option('--dataset', '-d', default=os.getenv('DATASET_PATH', ''), type=click.Path(exists=True), help='Path to the dataset')
def merge(dataset, report):
    r = pd.read_csv(report)
    r = r[r['Accept / Reject'] == 'Accept']
    d = pd.read_csv(dataset)

    # Merge the dataframes on 'query_id'
    merged = pd.merge(d, r[['query_id', 'actual_id', 'actual']], on='query_id', how='left')

    # Update 'expected_id' and 'expected_text' with 'actual_id' and 'actual'
    merged.loc[merged['actual_id'].notna(), 'expected_id'] = merged['actual_id']
    merged.loc[merged['actual'].notna(), 'expected_text'] = merged['actual']

    # Drop the 'actual_id' and 'actual' columns
    merged.drop(['actual_id', 'actual'], axis=1, inplace=True)

    # Update 'd' with 'merged'
    d = merged

    with Progress() as progress:

        task3 = progress.add_task("[cyan] Merging ...", total=merged.shape[0])
        while not progress.finished:
            progress.update(task3, advance=0.9)
            #time.sleep(0.01)
    
    console = Console()
    text = Text(str(merged.shape[0]) + " accepted answers have been merged into golden answers")
    text.stylize("bold magenta", 0, 6)
    console.print(text)
    d.to_csv('examples/merged_golden_dataset.csv', index=False)


def _generate_entry(id: int, q: str, ei: str, e: str, ai: str, a: str, mrr: float) -> Entry:
    return Entry(
        query_id=id,
        question=q,
        expected_id=ei,
        expected=e,
        actual_id=ai,
        actual=a,
        mrr=mrr  # Generate a random float between 0 and 1
    )

def _create_process_table(console, processes: list) -> Table:

    table = Table()
    table.add_column("ID", ratio=1)
    table.add_column("QUESTION", overflow="crop", no_wrap=True, ratio=2)
    table.add_column("EXPECTED",overflow="crop", no_wrap=True, ratio=2)
    table.add_column("ACTUAL", overflow="crop", no_wrap=True, ratio=2)
    table.add_column("MRR", style="dim", width=10, ratio=1)
    table.width = console.width * 2 // 3

    for process in processes:
        mrr_color = "green" if process.mrr > 0.7 else "yellow" if process.mrr > 0.3 else "red"
        mrr_text = Text(f"{process.mrr:.1f}", style=mrr_color)
        table.add_row(
            str(process.id),
            process.question,
            process.expected,
            process.actual,
            mrr_text,
        )

    return table

if __name__ == '__main__':
    evaluate()