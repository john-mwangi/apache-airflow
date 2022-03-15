from datetime import datetime, timedelta
from airflow.models import DAG

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

    # Task 1
    """
    Create database tables for storing training data and results of model training.
    """
