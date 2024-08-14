import os
import csv
from dotenv import load_dotenv
from pathlib import Path
from influxdb_client import InfluxDBClient
import pandas as pd

load_dotenv()  # Load the environment variables from the .env file in this root folder

# Assign the environment values to the variables
token = os.getenv('TOKEN')
host = os.getenv('HOST')
organization = os.getenv('ORGANIZATION')

# Create a client connection to the influx db server
client = InfluxDBClient(url=host, token=token, org=organization)

query_api = client.query_api()

# Main query that takes data from k6db_api bucket, range from the start time to stop time (UTC), and takes all the fields' value (NOT URL)
query = '''from(bucket: "k6db_api")
  |> range(start:2024-08-01T12:30:00Z, stop:2024-08-01T13:15:00Z)
  |> filter(fn: (r) => r["_measurement"] == "checks" or r["_measurement"] == "data_received" or r["_measurement"] == "emptyBody" or r["_measurement"] == "failed_EDI_API_requests" or r["_measurement"] == "data_sent" or r["_measurement"] == "failed_requests" or r["_measurement"] == "group_duration" or r["_measurement"] == "http_req_blocked" or r["_measurement"] == "http_req_connecting" or r["_measurement"] == "http_req_duration" or r["_measurement"] == "http_req_failed" or r["_measurement"] == "http_req_receiving" or r["_measurement"] == "http_req_sending" or r["_measurement"] == "http_req_tls_handshaking" or r["_measurement"] == "http_req_waiting" or r["_measurement"] == "http_reqs" or r["_measurement"] == "iteration_duration" or r["_measurement"] == "iterations" or r["_measurement"] == "slow_requests" or r["_measurement"] == "vus" or r["_measurement"] == "vus_max")
  |> filter(fn: (r) => r["_field"] == "value")'''

tables = query_api.query(query, org="Perf")

# Code below send the api query and write them into a csv file
headers = ["_value", "_field", "_measurement", "script"]  # Headers we want (There were columns like "result", "time start", "time" which are uselesss)
filtered_record = []  # to hold each row of data

df = None

# We basically go through the data and only takes the column values defined in "headers", then create a pandas DataFrame before throwing the content into a csv file
# We use pandas because of the speed it offers. Pure python code will take around 30-60 minutes while this take 5 minutes.

for i, table in enumerate(tables):
    for record in table.records:
        row = []
        for key in headers:
            try:
                row.append(record.values[key])
            except:
                row.append("")

        filtered_record.append(row)

df = pd.DataFrame(filtered_record, columns=headers)
filename = f"data.csv"

output_location = (Path(__file__).parent/"data"/filename)

Path(output_location).parents[0].mkdir(parents=True, exist_ok=True)  # Create the 'data' directory if not exist

df.to_csv(output_location)  # Write csv file

print("Query results written to CSV files.")  # To show that the csv file is produce successfully.