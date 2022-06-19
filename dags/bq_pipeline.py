from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from utils.utils import AirflowPipeline

ap = AirflowPipeline()

"""
Tasks:
- Set BigQuery credentials
- Import BigQuery table and convert to df using a Python function
- Print shape of df
- Write to another BigQuery table without transformation using a Python function
"""

default_args = {
    "email": ["alerts.mwangi@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(year=2022, month=3, day=21, hour=20, minute=13),
}

with DAG(
    dag_id="bq_pipeline",
    description="Reads a BQ Table as a Pandas dafaframe and writes to another BiqQuery Table",
    schedule_interval=timedelta(hours=4),
    default_args=default_args,
    catchup=False,
) as dag:
    
    # Task 2: print shape of df
    
    print_shape = PythonOperator(
        task_id="print_shape",
        python_callable=ap.print_df_shape
    )
    
    # Task 3: Import BigQuery table and convert to df
    
    read_bq_table = PythonOperator(
        task_id="read_bq_table",
        python_callable=ap.read_bq_table,
        op_kwargs={
            "tablename": "transactions_01",
            "datasetid": "fs_test01",
            "projectid": "ph-test-01",
        })
    
    # Task 4: Write to another BigQuery table without transformation
    
    write_bq_table = PythonOperator(
        task_id="write_bq_table",
        python_callable=ap.write_bq_table,
        op_kwargs={
            "bqprojectid": "ph-test-01",
            "datasetid": "fs_test01",
            "tablename": "transactions_02",
        })
    
    read_bq_table >> print_shape >> write_bq_table