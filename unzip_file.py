# python script to
# 1. download all data .gz files from GCS
# 2. decompress files and save locally as csv
# 3. read csv files into dataframe and check for errors
# 4. upload csv to GCS
# 5. upload dataframe to Bigquery

from google.cloud import storage
import os
import gzip
import shutil
from pathlib import Path
import pandas_gbq
import pandas as pd
import numpy as np
import logging
import re
import json
from fuzzywuzzy import process, fuzz
from dask.distributed import Client as dClient
import dask.dataframe as dd
from multiprocessing import Pool

# --------------------------
# to handle large csv fields
import sys
import csv

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)
# --------------------------

# Create a local dask cluster with workers in same process
# dask_client = dClient(processes=False)

# Instantiates a client
storage_client = storage.Client()

project_id = "gum-eroski-dev"
dataset_id = "source_data"
bucket = "erk-data-feed"
blobs = storage_client.list_blobs(bucket, prefix="Working_folder/AT/ETL_test/")

blob_list = [blob.name for blob in blobs]
blob_fname = [blob.split("/")[-1] for blob in blob_list]
# print(blob_list)

home = str(Path.home())
local_dir = os.path.abspath(home + "/etl_test/")
# local_directory = os.fsencode("~/etl_test/")

# open table_mapping.json
with open("table_mapping.json", "r") as mapping:
    table_mapping = json.load(mapping)


def initialise_logger():
    """Initialise logger settings"""

    # Set logger properties
    logger = logging.getLogger("auto_etl_to_bq")
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler("auto_etl_to_bq.log")
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    logger.info("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def gunzip(source_filepath, dest_filepath, block_size=65536):
    """Unzips .gz files and writes to persistent disk"""

    with gzip.open(source_filepath, "rb") as s_file, open(dest_filepath, "wb") as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)
        d_file.write(block)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket"""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logger.info("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def change_extension(old_extension, new_extension, directory):
    """Change file extensions for files in directory"""

    for file in os.listdir(directory):
        pre, ext = os.path.splitext(file)
        if ext == old_extension:
            os.rename(file, pre + new_extension)
            continue
        else:
            continue


def csv_checks(csv_filename, dataset_schema):
    """Checks format of delta csv files with Bigquery tables"""

    logger.info("-------------Beginning checks for {}-------------".format(csv_filename))
    # read csv file into dataframe
    try:
        csv_data = pd.read_csv(
            csv_filename, header=None, index_col=False, sep="|", engine="python", nrows=10
        )
        full_csv_data = dd.read_csv(
            csv_filename,
            header=None,
            sep="|",
            engine="python",
            assume_missing=True,
            dtype="str",
            quotechar='"',
            error_bad_lines=False,
        )
        logger.info("csv file: {} loaded to dataframe".format(csv_filename))
        # logger.info(csv_data.head())
        # logger.info("first index value is {}".format(full_csv_data.head().index[0]))
        # if index is not default index reset index and drop last column
        if full_csv_data.head().index[0] != 0:
            full_csv_data = full_csv_data.reset_index()
            full_csv_data = full_csv_data.iloc[:, :-1]
        # logger.info(full_csv_data.head())
        logger.info("number of partitions = {}".format(full_csv_data.npartitions))
        read_successful = True
    except:
        logger.info("csv file: {} did not read properly".format(csv_filename))
        read_successful = False

    # csv_data = dd.read_csv(csv_filename, header=None, sep="|", engine="python", assume_missing=True)
    # check csv dataframe is not empty
    # if csv_data.empty == False:
    if read_successful:
        # logger.info(csv_data.describe(include="all"))
        # check for matching table in Bigquery
        fn = csv_filename.split("/")[-1]
        table_name_list = dataset_schema.table_name.unique()

        # replace digits with x and remove extension
        fn_str = re.sub(r"\d", "X", fn.split(".")[0])

        # if file mapping exists for file name
        if table_mapping[fn_str] != "":
            # using mapping table select the correct schema
            matched_table_schema = dataset_schema.loc[
                dataset_schema.table_name == table_mapping[fn_str]
            ]
            # get first row of csv dataframe
            csv_header = list(csv_data.iloc[0])
            # logger.info(csv_header)
            # get column names of bq table
            table_columns = matched_table_schema.column_name.tolist()
            # logger.info(table_columns)
            # compare csv headers and column names
            csv_header = [str(x).lower() for x in csv_header]
            table_columns_lower = [x.lower() for x in table_columns]
            if len(csv_header) == len(table_columns_lower) and len(csv_header) == sum(
                [1 for i, j in zip(csv_header, table_columns_lower) if i == j]
            ):
                logger.info("HEADERS MATCHED")
                # use first row as header and drop
                full_csv_data.columns = table_columns
                csv_data.columns = table_columns
                full_csv_data = full_csv_data.loc[1:]
                csv_data = csv_data.iloc[1:]
            elif len(csv_header) == len(table_columns):
                # same csv header count and bq table column count
                logger.info("Adding headers to {}".format(fn))
                # add bq table column as header
                full_csv_data.columns = table_columns
                csv_data.columns = table_columns
                # logger.info(csv_data.head())
            else:
                # not matched - error
                logger.info("Headers do not match")
                # add bq table column as headers
                # logger.info(csv_data.head())
                logger.info("csv has {a} columns".format(a=len(csv_header)))
                logger.info("bq table has {b} columns".format(b=len(table_columns)))
                # add blank columns missing from bq table to csv dataframe
                for c in range(1, len(table_columns) + 1 - len(csv_header)):
                    full_csv_data["new_column_{}".format(c)] = np.nan
                    csv_data["new_column_{}".format(c)] = np.nan
                    logger.info("empty column added!")
                # logger.info(csv_data.head())
                assert full_csv_data.shape[1] == len(table_columns)
                # add bq table column as headers
                full_csv_data.columns = table_columns
                csv_data.columns = table_columns
                # logger.info(csv_data.head())

                # logger.info("Did not attempt to upload {} to Bigquery".format(fn))
            if csv_data[csv_data.columns[0]].iloc[0] == csv_data.columns[0]:
                # first row is the same as header
                logger.info("dropping first row")
                # logger.info(full_csv_data.head())
                full_csv_data = full_csv_data.loc[1:]
                csv_data = csv_data.iloc[1:]
                # logger.info(full_csv_data.head())
                # logger.info(csv_data.head())
            else:
                # logger.info(csv_data.head())
                logger.info("matched bq table")

            # clean up csv_data
            # remove duplicates
            full_csv_data.drop_duplicates(inplace=True)
            # reset index
            full_csv_data = full_csv_data.reset_index(drop=True)
            # add timestamp column
            full_csv_data["timestamp"] = re.findall("\d+", fn)
            # remove quotation marks
            # full_csv_data = full_csv_data.map_partitions(lambda d: d.replace('"', ""))
            # csv_data = csv_data.compute()
            try:
                logger.info(full_csv_data.head())
            except:
                logger.info("Could not parse csv")
                # logger.info(csv_data.head())
        else:
            logger.info("Delta table {} does not have mapping".format(fn))
    else:
        logger.info("Did not attempt to upload {} to Bigquery".format(csv_filename))


def get_bq_schemas(dataset_id):
    """Returns Bigquery dataset information"""

    # get table names and columns
    sql_str = """
    SELECT table_name, column_name, data_type FROM `gum-eroski-dev`.source_data.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS
    """
    # read from Bigquery
    dataset_schema = pandas_gbq.read_gbq(sql_str, project_id=project_id)
    return dataset_schema


if __name__ == "__main__":
    logger = initialise_logger()
    dataset_schema = get_bq_schemas(dataset_id)
    for blob in blob_list:
        blob_fn = blob.split("/")[-1]
        # check if file exists
        if os.path.exists(os.path.abspath(local_dir + "/" + blob_fn)):
            logger.info(
                "Blob {} already exists in {}.".format(
                    blob, os.path.abspath(local_dir + "/" + blob_fn)
                )
            )
        else:
            download_blob(bucket, blob, os.path.abspath(local_dir + "/" + blob_fn))
        if os.path.exists(os.path.abspath(local_dir + "/" + blob_fn.split(".")[0] + ".csv")):
            logger.info(
                "File {} already unzipped".format(os.path.abspath(local_dir + "/" + blob_fn))
            )
            csv_checks(
                os.path.abspath(local_dir + "/" + blob_fn.split(".")[0] + ".csv"), dataset_schema
            )
        else:
            gunzip(
                os.path.abspath(local_dir + "/" + blob_fn),
                os.path.abspath(local_dir + "/" + blob_fn.split(".")[0] + ".csv"),
            )
        # upload_blob(
        #     bucket,
        #     os.path.abspath(local_dir + "/" + blob_fn.split(".")[0] + ".csv"),
        #     "Working_folder/AT/ETL_test_upload/" + blob_fn.split(".")[0] + ".csv",
        # )

        # upload dataframe to Bigquery
        # pandas_gbq.to_gbq(blob_dataframe, )
