import os
import json
from pathlib import Path
from google.cloud import bigquery

client = bigquery.Client()

home = str(Path.home())
filename = os.path.abspath(home + "/baseline/test.csv")
dataset_id = "WIP"
table_id = "upload_test"

dataset_ref = client.dataset(dataset_id)
table_ref = dataset_ref.table(table_id)
job_config = bigquery.LoadJobConfig()
job_config.source_format = bigquery.SourceFormat.CSV
job_config.autodetect = True

job_config.skip_leading_rows = 1
job_config.field_delimiter = ","

s = """{
  "fields": [
    {
      {
        "name": "one",
        "type": string
      },
      {
        "name": "two",
        "type": string
      },
      {
        "name": "three",
        "type": string
      },
      {
        "name": "four",
        "type": string
      }
    }
  ]
}"""
schema = json.loads(s)
job_config.allow_jagged_rows = True
job_config.schema = schema

with open(filename, "rb") as source_file:
    job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

job.result()  # Waits for table load to complete.
