from google.cloud.bigquery import SchemaField

TRAIN_SIZE = 0.7
RANDOM_SEED = 123
DB_STRING = "postgresql+psycopg2://airflow:airflow@postgres/airflow"
MAX_PCA_COMPONENTS = 30
CV_FOLDS = 5
MAX_ITER = 100
OUT_PATH = "/opt/airflow/models/"
# DATA_PATH = "../../data/"
# DATA_PATH = "/opt/airflow/data/"
DATA_PATH = "/home/mail_mwangi/airflow/data/"
TABLE_SCHEMA = [
    SchemaField("updatedBy", "STRING"),
    SchemaField("status", "STRING"),
    SchemaField("customerId", "STRING"),
    ]