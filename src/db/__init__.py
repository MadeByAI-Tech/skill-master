from google.api_core.client_options import ClientOptions
from google.auth.credentials import AnonymousCredentials
from google.cloud import bigquery

import os

BQ_PROJECT = os.environ["BQ_PROJECT"]
BQ_DATASET = os.environ["BQ_DATASET"]

CREDENTIAL = AnonymousCredentials() 
_client: bigquery.Client
dataset_id: str

if os.environ["MODE"] == "dev":
    BQ_HOST = os.environ["BQ_HOST"]
    BQ_PORT = os.environ["BQ_PORT"]
    _client_options = ClientOptions(api_endpoint=f"http://{BQ_HOST}:{BQ_PORT}")
    _client = bigquery.Client(
    project=BQ_PROJECT,
    client_options=_client_options,
    credentials=CREDENTIAL,
    )
    dataset_id = f"{_client.project}.{BQ_DATASET}"
elif os.environ["MODE"] == "staging":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.environ["BQ_SA_FROM"]
    dataset_id = f"{BQ_PROJECT}.{BQ_DATASET}"
else:
    raise NotImplementedError(f"MODE={os.environ['MODE']} not implemented")


def get_dataset() -> bigquery.Dataset:
    dataset = bigquery.Dataset(dataset_id)
    try:
        dataset = _client.get_dataset(dataset_id)
        print("Dataset already exists")
    except Exception:
        print("Dataset not found: creating...")
        dataset = _client.create_dataset(dataset, exists_ok=True)
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