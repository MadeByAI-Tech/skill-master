from pydantic import BaseModel
from google.cloud import bigquery
from typing import Any
import json

"""
Schema model for `JobPostExtracted` table, which stores processed job postings after LLM extraction.

Provides:
    - get_schema()        : Defines `JobPostExtracted` table schema
    - create_table()      : Creates `JobPostExtracted` table
    - delete_table()      : Deletes `JobPostExtracted` table
    - fetch_all()         : Reads all rows from `JobPostExtracted` table
    - insert()            ; Inserts a single record into `JobPostExtracted` table
"""

class JobPostExtracted(BaseModel):
    job_id: int
    job_title: str
    location: str
    job_function: str
    description: str
    skills: list[dict[str, Any]]
    cwf_items: list[dict[str, Any]]

    @classmethod
    def get_schema(cls):
        return [
            bigquery.SchemaField("job_id",          "INTEGER",  mode="REQUIRED"),
            bigquery.SchemaField("job_title",       "STRING",   mode="REQUIRED"),
            bigquery.SchemaField("location",        "STRING",   mode="REQUIRED"),
            bigquery.SchemaField("job_function",    "STRING",   mode="REQUIRED"),
            bigquery.SchemaField("description",     "STRING",   mode="REQUIRED"),
            bigquery.SchemaField("skills",          "JSON",     mode="REQUIRED"),
            bigquery.SchemaField("cwf_items",       "JSON",     mode="REQUIRED"),
            # bigquery.SchemaField("version_id", "INTEGER", mode="REQUIRED"),
        ]

    @classmethod
    def create_table(cls, dataset_id: str, client: bigquery.Client):
        schema = cls.get_schema()
        table_id = f"{client.project}.{dataset_id}.JobPostExtracted"
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table, exists_ok=True)
        print(f"Created `{table_id}` table")

    @classmethod
    def delete_table(cls, dataset_id: str, client: bigquery.Client):
        table_id = f"{client.project}.{dataset_id}.JobPostExtracted"
        client.delete_table(table_id, not_found_ok=True)
        print(f"Deleted ` {table_id}` table")

    @classmethod
    def fetch_all(cls, dataset_id: str, client: bigquery.Client) -> list["JobPostExtracted"]:
        query = f"""
            SELECT *
            FROM `{dataset_id}.JobPostExtracted`
        """
        rows = client.query_and_wait(query)

        results: list[cls] = []
        for r in rows:
            results.append(
                JobPostExtracted(
                    job_id=r["job_id"],
                    job_title_cleaned=r["job_title_cleaned"],
                    job_level=r["job_level"],
                    skills=r["skills"],
                    cwf_items=r["cwf_items"],
                )
            )
        return results

    def insert(self, dataset_id: str, client: bigquery.Client):
        errors = client.insert_rows_json(
            f"{dataset_id}.JobPostExtracted",
            [json.loads(self.model_dump_json())]
        )
        if errors:
            print("Insert errors:", errors)