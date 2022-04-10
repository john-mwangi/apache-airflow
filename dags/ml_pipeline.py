from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.postgres_operator import PostgresOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.task_group import TaskGroup

from utils.utils import AirflowPipeline

ap = AirflowPipeline()

"""
Pseudo-code:
- Import sklearn datasets and convert to df
- Split into train and test sets
- Train model (scale, pca, logistic regression)
- Save training results to db
- Save training data to db
"""

# default_args = https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html#default-arguments

default_args = {
    "email": ["alerts.mwangi@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(year=2022, month=3, day=21, hour=20, minute=13),
}

with DAG(
    dag_id="ml_pipeline",
    description="End to end ML pipeline",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
) as dag:

    # Task 1: Create database tables for storing training data and results of model training.

    with TaskGroup(group_id="create_tables") as create_tables:
        # Task 1.1 training data
        create_training_data_table = PostgresOperator(
            task_id="create_training_data_table",
            postgres_conn_id="postgres",
            sql="sql/create_training_data_table.sql",
        )

        # Task 1.2 training results
        create_training_results_table = PostgresOperator(
            task_id="create_training_results_table",
            postgres_conn_id="postgres",
            sql="sql/create_training_results_table.sql",
        )

    # Task 2: Use a python function to load data into database.

    load_data = PythonOperator(
        task_id="load_data", python_callable=ap.load_data
    )

    # Task 3: Preprocess and save data to database

    with TaskGroup(group_id="prepare_data") as prep_data:
        # Task 3.1
        preprocess_data = PythonOperator(
            task_id="preprocess_data",
            python_callable=ap.preprocess_data,
        )

        # Task 3.2
        PythonOperator(task_id="save_data", python_callable=ap.save_data)

    # Task 4: Train model

    run_train = PythonOperator(
        task_id="run_train",
        python_callable=ap.run_train,
    )

    # Task 5: Save training results and model
    with TaskGroup(group_id="save_results") as save_results:
        # Task 5.1
        save_training_info = PythonOperator(
            task_id="save_training_info", python_callable=ap.save_training_info
        )

        # Task 5.2
        save_model = PythonOperator(
            task_id="save_model",
            python_callable=ap.save_model,
        )

    create_tables >> load_data >> prep_data >> run_train >> save_results
