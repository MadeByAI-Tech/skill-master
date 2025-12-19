from pydantic import BaseModel, field_validator
from datetime import datetime
from google.cloud import bigquery
from typing import Any
import json

class JobPost(BaseModel):
    job_id: str
    url: str
    work_type: str
    salary: str
    description: str
    posted_text: str
    postedAt: datetime

    @field_validator('postedAt', mode='before')
    @classmethod
    def parse_datetime(cls, value: str) -> datetime:
        if(isinstance(value, datetime)):
            return value    
        # 2025-07-30 18:03:10.961148 UTC
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f UTC")
        return dt
    
    @classmethod
    def get_schema(cls):
        schema = [
            bigquery.SchemaField("job_id",     "STRING",    mode="REQUIRED"),
            bigquery.SchemaField("url",        "STRING",    mode="REQUIRED"),
            bigquery.SchemaField("work_type",  "STRING",    mode="REQUIRED"),
            bigquery.SchemaField("salary",     "STRING",    mode="REQUIRED"),
            bigquery.SchemaField("description","STRING",    mode="REQUIRED"),
            bigquery.SchemaField("posted_text","STRING",    mode="REQUIRED"),
            bigquery.SchemaField("postedAt",   "TIMESTAMP", mode="REQUIRED"),
        ]
        return schema

    @classmethod
    def create_table(cls, dataset_id: str, client: bigquery.Client):
        schema = cls.get_schema()
        table_id = f"{dataset_id}.JobPosts"
        table = bigquery.Table(table_id, schema=schema)
        table = client.create_table(table)  # Make an API request.
        print(
            "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
        )

    @classmethod
    def delete_table(cls, dataset_id: str, client: bigquery.Client):
        client.delete_table(f"{dataset_id}.JobPosts", not_found_ok=True)
        print("Deleted table '{}'.".format("JobPosts"))
    
    @classmethod
    def fetch_all(cls, dataset_id: str, client: bigquery.Client) -> list["JobPost"]:
        query = f"""
            SELECT *
            FROM {dataset_id}.JobPosts
        """
        rows = client.query_and_wait(query)  # Make an API request.
        results:list[cls] = []
        for row in rows:
            job_post = JobPost(
                job_id = row["job_id"],
                url = row["url"],
                work_type = row["work_type"],
                salary = row["salary"],
                description = row["description"],
                posted_text = row["posted_text"],
                postedAt = row["postedAt"]
            )
            results.append(job_post)
        return results

    def insert(self, dataset_id: str, client: bigquery.Client):
        errors:[Any] = client.insert_rows_json(f"{dataset_id}.JobPosts", [json.loads(self.model_dump_json())])  # Make an API request.
        if errors != []:
            print(f"Encountered errors while inserting rows: {errors}")

    