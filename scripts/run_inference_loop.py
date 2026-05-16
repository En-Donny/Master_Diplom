import pandas as pd
from influx.influxdb import InfluxDBProcesser
from model_PINN.inference_mlflow import InferencePINNModel
from utils.logger import get_logger
import mlflow
import time

def set_tracking():
    # Connect to remote MLflow server
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("my-first-PINN-experiment")


def main():
    logger = get_logger("INFERENCE_LOOP")
    try:
        with open("/run/secrets/influxdb2_admin_token", "r", encoding="utf-8") as file:
            token = file.read().strip()
        infl_db_class = InfluxDBProcesser(url="http://influxdb2:8086", token=token)
        infl_db_class.connect()
        logger.info("Successfully connected to InfluxDB!")
        check_last_time = infl_db_class.get_timestamp(measurement="test_measurement", mode="last")
        logger.info(f"The last time got: {check_last_time}")

        if not check_last_time:
            logger.info("Start process of loading initial data...")
            base_data = pd.read_csv("./influx/syntetic_data_for_influx.csv", parse_dates=True, index_col="Timestamp")
            infl_db_class.insert_data_from_df(base_data, measurement="test_measurement")
            logger.info("Successfully loaded initial data!")

            try:
                set_tracking()
                inf_pinn = InferencePINNModel()
                with mlflow.start_run():
                    model_info = mlflow.pyfunc.log_model(
                        artifact_path="custom_pinn_model",
                        python_model=inf_pinn,
                        registered_model_name="CustomPINNModel",
                    )
                    with open("model_mlflow_uri.txt", "w", encoding="utf-8") as uri:
                        uri.write(model_info.model_uri)
                logger.info("PINN successfully loged!")
            except Exception as e:
                logger.info(f"Something went wrong while connecting to MLFlow or setting experiment. Error: {e}")

    except Exception as e:
        logger.info(f"Something went wrong while connecting to InfluxDB and loading initial data. Error: {e}")
        raise e

    with open("model_mlflow_uri.txt", "r", encoding="utf-8") as uri:
        model_pinn_uri = uri.read().strip()
    while True:
        loaded_pinn_model = mlflow.pyfunc.load_model(model_pinn_uri)
        new_data = infl_db_class.inference_data_step(measurement="test_measurement")
        test_infl_db = loaded_pinn_model.predict(new_data)
        infl_db_class.insert_data_from_df(test_infl_db, measurement="model_predicts")
        logger.info("Successfully got new predictions from PINN!")
        time.sleep(30)
        del loaded_pinn_model



if __name__ == "__main__":
    main()