from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import influxdb_client
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta
from utils.logger import get_logger
import io
import logging


logger = get_logger("INFLUXDB_LOGS")

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


class InfluxDBProcesser:

    def __init__(self,
                 token:str = None,
                 url:str = "http://localhost:8086",
                 org:str = "Diplom",
                 bucket:str = "Diplom_signals_bucket"):
        
        self.token = token if token else "YOUR_TOKEN"
        self.url = url
        self.org = org
        self.bucket = bucket

        self.features_global_random_mean_and_var = {
            "I": (50.3, 8.0, 27, 80),
            "Q": (92.3, 30.5, 30, 200),
            "n": (70, 13.5, 44, 100),
            "x_1": (0.4, 0.065, 0.25, 0.6),
            "x_2": (27.5, 6.5, 7, 38),
            "x_3": (15.5, 10.5, 0, 65)
        }

    def connect(self):
        self.client = influxdb_client.InfluxDBClient(
            url=self.url, token=self.token, org=self.org
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

    def dissconnect(self):
        self.client.close()

    def get_syntetic_data(self, start_time: str) -> pd.DataFrame:
        test_index = pd.date_range(start=start_time, periods=10, freq="30s")
        fut_data = {}
        for col in self.features_global_random_mean_and_var.keys():
            mean, var, min, max = self.features_global_random_mean_and_var[col]
            std = np.sqrt(var)
            fut_data[col] = np.clip(
                np.random.normal(mean, std, len(test_index)), min, max
            )
        return pd.DataFrame(index=test_index, data=fut_data)


    def clear_entries(self,
                      start_date: str,
                      end_date: str,
                      measurement:str) -> None:
        """
        Clear all entries in Influx from start_date to end_date.
        """
        try:
            delete_api = self.client.delete_api()    
            delete_api.delete(start_date,
                              end_date,
                              f'_measurement="{measurement}"',
                              bucket=self.bucket,
                              org=self.org)
        except Exception as e:
            logger.info(f"Something went wrong while clearing data. Error: [{e}]")

    def get_timestamp(self, measurement:str, mode="last") -> datetime:
        """
        Get timestemp of first|last entry in InfluxDB.
        """
        """
        Get timestamp of last entry in data.
        """
        query = f"""from(bucket: "{self.bucket}")
            |> range(start: 2026-01-01T08:00:00Z, stop: 2026-05-30T08:00:00Z)
            |> filter(fn: (r) => r._measurement == "{measurement}")"""
        extr = "\n|> tail(n:1)"
        if mode == "first":
            extr = "\n|> limit(n:1)"
        query += extr
        try:
            res_tables = self.query_api.query(query, org=self.org)
            last_timestamp = None
            for table in res_tables:
                for record in table.records:
                    last_timestamp = record.get_time()
                    return last_timestamp
        except Exception as e:
            logger.info(f"Something went wrong while getting last timestamp. Error: [{e}]")
            return None

    def insert_data_from_df(self, df: pd.DataFrame, measurement:str):
        """
        Insert into InfluxDB data from argument dataframe.
        """
        try:
            cols = df.columns
            tqdm_out = TqdmToLogger(logger, level=logging.INFO)
            for index in tqdm(df.index, file=tqdm_out, mininterval=30,):
                cur_row = df.loc[index]
                for col in cols:
                    point = (
                        Point(measurement)
                        .tag("tagname1", "tagvalue1")
                        .field(col, cur_row[col])
                        .time(datetime.strftime(index, "%Y-%m-%dT%H:%M:%SZ"))
                    )
                    self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            del tqdm_out
        except Exception as e:
            logger.info(f"Something went wrong while inserting data. Error: [{e}]")

    def get_last_hour_data(self,
                           last_timestamp: datetime,
                           measurement:str) -> pd.DataFrame | None:
        """
        Get last one hour of data from InfluxDB.
        """
        try:
            start_time = datetime.strftime(last_timestamp - timedelta(hours=1),
                                           "%Y-%m-%dT%H:%M:%SZ")
            end_time = datetime.strftime(last_timestamp + timedelta(minutes=1),
                                         "%Y-%m-%dT%H:%M:%SZ")
            query = f"""from(bucket: "{self.bucket}")
                |> range(start: {start_time}, stop: {end_time})
                |> filter(fn: (r) => r._measurement == "{measurement}")"""

            tables = self.query_api.query(query, org=self.org)
            data = defaultdict(list)
            index_list = []
            for table_ind, table in enumerate(tables):
                for record in table.records:
                    if not table_ind:
                        index_list.append(record.get_time())
                    field_name = record.get_field()
                    data[field_name].append(record.get_value())
            
            return pd.DataFrame(index=index_list, data=data)
        except Exception as e:
            logger.info(f"Something went wrong while getting last 1 hour data. Error: [{e}]")

    def inference_data_step(self, measurement:str):
        """
        Get data for one step of inference.
        """
        first_time = self.get_timestamp(measurement=measurement, mode="first")
        self.clear_entries(
            datetime.strftime(first_time, "%Y-%m-%dT%H:%M:%SZ"),
            datetime.strftime(first_time + timedelta(minutes=4, seconds=30),
                              "%Y-%m-%dT%H:%M:%SZ"),
            measurement=measurement,
        )
        logger.info("Successfully deleted first 5 minutes of data!")

        last_time = self.get_timestamp(measurement=measurement, mode="last")
        insert_data = self.get_syntetic_data(
            datetime.strftime(last_time + timedelta(seconds=30),
                              "%Y-%m-%dT%H:%M:%S")
        )
        self.insert_data_from_df(insert_data, measurement=measurement)
        logger.info("Successfully inserted new 5 minutes of data to DB!")
        last_time = self.get_timestamp(measurement=measurement, mode="last")

        return self.get_last_hour_data(last_time, measurement=measurement)