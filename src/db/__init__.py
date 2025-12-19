from google.api_core.client_options import ClientOptions
from google.auth.credentials import AnonymousCredentials
from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig

import os

# BQ_HOST=bigquery
# BQ_PORT=9050
# BQ_PROJECT=skill-master
# BQ_DATASET=dev
BQ_HOST = os.environ["BQ_HOST"]
BQ_PORT = os.environ["BQ_PORT"]
BQ_PROJECT = os.environ["BQ_PROJECT"]
BQ_DATASET = os.environ["BQ_DATASET"]

CREDENTIAL = AnonymousCredentials() 
if os.environ["MODE"] != "dev":
    raise NotImplementedError("Only dev mode is supported now in bigquery module.")

_client_options = ClientOptions(api_endpoint=f"http://{BQ_HOST}:{BQ_PORT}")
_client = bigquery.Client(
  project=BQ_PROJECT,
  client_options=_client_options,
  credentials=CREDENTIAL,
)
dataset_id = f"{_client.project}.{BQ_DATASET}"

def get_dataset() -> bigquery.Dataset:
    dataset = bigquery.Dataset(dataset_id)
    dataset = _client.create_dataset(dataset, timeout=30, exists_ok=True)  # Make an API request.
    # print("Created dataset {}.{}".format(_client.project, dataset.dataset_id))
    return dataset

def get_table(table_name: str) -> bigquery.Table:
    _ = get_dataset()
    table_id = f"{dataset_id}.{table_name}"
    table = bigquery.Table(table_id)
    table = _client.create_table(table, exists_ok=True)  # Make an API request.
    # print("Created table {}.{}.{}".format(client.project, dataset.dataset_id, table.table_id))
    return table

def get_client() -> bigquery.Client:
    return _client