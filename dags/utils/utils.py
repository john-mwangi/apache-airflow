import os
from datetime import datetime, timedelta

import dill
import numpy as np
import pandas as pd
from google.cloud import bigquery

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

import pytz

# import config


class AirflowPipeline:
    @staticmethod
    def save_files(files: dict):
        """Save a list of objects as pickle files.

        files: dictionary of objects to save
        """
        for file_name, object in files.items():
            with open(
                file=os.path.join(config.DATA_PATH, f"{file_name}.bin"),
                mode="wb",
            ) as f:
                dill.dump(obj=object, file=f)

    @staticmethod
    def load_files(files: list):
        """Load a list of objects from pickle files.

        files: list of file names to load
        """
        for item in files:
            with open(
                file=os.path.join(config.DATA_PATH, f"{item}.bin"), mode="rb"
            ) as f:
                yield dill.load(file=f)

    def load_data(self):
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        self.save_files(files={"X": X, "y": y})

    def preprocess_data(self):
        X, y = self.load_files(files=["X", "y"])
        train_X, test_X, train_y, test_y = train_test_split(
            X,
            y,
            train_size=config.TRAIN_SIZE,
            random_state=config.RANDOM_SEED,
        )

        self.save_files(
            files={
                "train_X": train_X,
                "test_X": test_X,
                "train_y": train_y,
                "test_y": test_y,
            }
        )

    def save_data(self):
        """Save raw data to db."""

        X, y = self.load_files(files=["X", "y"])
        data = pd.concat(objs=[X, y], axis=1)
        data.to_sql(
            name="training_batch",
            con=config.DB_STRING,
            if_exists="replace",
            index=False,
        )

    def run_train(self):
        train_X, test_X, train_y, test_y = self.load_files(
            files=["train_X", "test_X", "train_y", "test_y"]
        )

        # Training parameters
        max_pca = (
            train_X.shape[1]
            if config.MAX_PCA_COMPONENTS >= train_X.shape[1]
            else config.MAX_PCA_COMPONENTS
        )
        cv_folds = config.CV_FOLDS
        max_iter = config.MAX_ITER

        # Define pipeline
        scaler = StandardScaler()
        pca = PCA(n_components=max_pca)
        log_reg = LogisticRegression(max_iter=max_iter)

        pipe = Pipeline(
            steps=[("scaler", scaler), ("pca", pca), ("log_reg", log_reg)]
        )

        # Cross validated training
        param_dist = {
            "pca__n_components": np.arange(1, max_pca),
            "log_reg__C": np.logspace(start=0.05, stop=5, num=10, base=10),
        }

        log_reg_cv = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            cv=cv_folds,
            random_state=config.RANDOM_SEED,
            scoring="roc_auc",
        )

        log_reg_cv.fit(X=train_X, y=train_y)
        ts = datetime.now().timestamp()

        # Retrieve best params
        best_penalty = log_reg_cv.best_params_.get("log_reg__C")
        best_num_comp = log_reg_cv.best_params_.get("pca__n_components")
        model = log_reg_cv.best_estimator_

        print("\n")
        print(f"best_penalty: {best_penalty}")
        print(f"best_num_comp: {best_num_comp}")

        # Performance on test
        preds = model.predict(test_X)
        auc = roc_auc_score(y_true=test_y, y_score=preds)
        print(f"AUC: {auc}")

        # Save training info to postgres
        ## timestamp, train_params, train_results (best_params), test_performance
        train_results = pd.DataFrame(
            data={
                "timestamp_utc": ts,
                "train_size": config.TRAIN_SIZE,
                "max_pca_components": max_pca,
                "cv_folds": config.CV_FOLDS,
                "max_iter": config.MAX_ITER,
                "best_penalty": best_penalty,
                "best_pca_components": best_num_comp,
                "auroc": auc,
            },
            index=[0],
        )

        self.save_files(
            files={"train_results": train_results, "model": model, "ts": ts}
        )

    def save_training_info(self):
        train_results = self.load_files(files=["train_results"])
        train_results = list(train_results)[0]
        train_results.to_sql(
            name="training_results",
            con=config.DB_STRING,
            if_exists="append",
            index=False,
        )

    def save_model(self):
        model, ts = self.load_files(files=["model", "ts"])
        with open(
            file=os.path.join(config.OUT_PATH, f"model_{ts}.bin"), mode="wb"
        ) as f:
            dill.dump(obj=model, file=f)

    def read_bq_table(
        self,
        ti,
        tablename: str,
        datasetid: str,
        projectid: str,
        clean: bool = False,
        primary_key: str = "",
    ) -> pd.DataFrame:
        """Fetches a Big Query table."""

        project_ref = bigquery.Client(project=projectid)
        sql_query = f"SELECT * from {datasetid}.{tablename}"
        res = project_ref.query(sql_query)
        rows = res.result()

        df = rows.to_dataframe()
        if clean:
            df = df.drop_duplicates(subset=primary_key).dropna(
                subset=primary_key
            )

        # self.save_files({"dataframe": df})
        ti.xcom_push(key="dataframe", value=df.to_dict("records"))

    def print_df_shape(self, ti):
        df_dict = ti.xcom_pull(task_ids="read_bq_table", key="dataframe")
        df = pd.DataFrame(df_dict)
        # df = self.load_files(files=["dataframe"])
        # df = list(df)[0]
        print(df.shape)

    def write_bq_table(
        self,
        ti,
        bqprojectid: str,
        datasetid: str,
        tablename: str,
        table_schema=config.TABLE_SCHEMA,
        truncate: bool = True,
    ):
        """Writes a pandas dataframe to BigQuery.

        Args:
        ----
        truncate: bool, if True, truncates the BigQuery table before writing. If False, appends.
        """

        df_dict = ti.xcom_pull(
            key="dataframe",
            task_ids="read_bq_table",
        )

        df = pd.DataFrame(df_dict)
        # df = self.load_files(files=["dataframe"])
        # df = list(df)[0]

        client = bigquery.Client()
        tableid = f"{bqprojectid}.{datasetid}.{tablename}"

        if truncate:
            res = client.query(f"TRUNCATE TABLE {tableid}")
            res.result()
            client.load_table_from_dataframe(dataframe=df, destination=tableid)
        else:
            errors = client.insert_rows_from_dataframe(
                table=tableid, dataframe=df, selected_fields=table_schema
            )

            for error in errors:
                if error:
                    print("Error:", error)

        print(f"Insertion complete.")


class LoanBook:
    DUE_DATE = datetime.now()
    EXCLUDE_PERIOD = int(DUE_DATE.strftime("%Y%m"))

    analysis_cols = [
        "loanId",
        "loan_due",
        "loanAmount",
        "category_90",
        "status",
        "amount_collected",
        "principal_due",
        "isRepaid",
        "isOutstanding",
        "disbursed_ym",
    ]

    col_names = {
        "disbursed_ym": "Period",
        "loanAmount": "Total Disbursed Amount",
        "outstanding_loanAmount": "Outstanding Amount",
        "outstanding_loan_count": "Outstanding Loans",
        "active_loanAmount": "Active Outstanding Amount",
        "active_loan_count": "Active Outstanding Loans",
    }

    default_args = {
        "email": ["alerts.mwangi@gmail.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
        "start_date": datetime(year=2022, month=3, day=21, hour=20, minute=13),
    }
    # GCP Key
    airflow_gcp_key = "/Users/johnmwangi/Work/payhippo/keys/adc/application_default_credentials.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = airflow_gcp_key

    def read_bq_table(
        tablename: str,
        datasetid: str,
        projectid: str = "payhippo",
        clean: bool = False,
        primary_key: str = None,
    ) -> pd.DataFrame:
        """Fetches a Big Query table."""

        project_ref = bigquery.Client(project=projectid)
        sql_query = f"SELECT * from {datasetid}.{tablename}"
        res = project_ref.query(sql_query)
        rows = res.result()

        df = rows.to_dataframe()
        if clean:
            if not primary_key:
                raise ValueError("provide a primary key")
            df = df.drop_duplicates(subset=primary_key).dropna(
                subset=primary_key
            )

        return df

    def calculate_principal_due(repaymentSchedules: pd.DataFrame) -> list:
        """Calculates the principal due for each repayment schedule.
        A helper function to add_payments_due_collected().\n
        if principalPayment is not NA, then principal_due is principalPayment\n
        if principalPayment is NA,then the principal due is the loanAmount when the paymentType is INTEREST_AND_PRINCIPAL_PAYMENT\n
        else, it is 0

        Args:
        ----
        repaymentSchedules: repaymentSchedules_proc - repaymentSchedules ready for analysis.
        """
        principal_due = []
        for principalPayment, loanAmount, paymentType in zip(
            repaymentSchedules.principalPayment,
            repaymentSchedules.loanAmount,
            repaymentSchedules.paymentType,
        ):
            if not pd.isna(principalPayment):
                principal_due.append(principalPayment)
            elif pd.isna(principalPayment) & (
                paymentType == "INTEREST_AND_PRINCIPAL_PAYMENT"
            ):
                principal_due.append(loanAmount)
            elif paymentType == "INTEREST_AND_ADMIN_FEE_PAYMENT":
                principal_due.append(0)
            else:
                raise ValueError(
                    "error on:", principalPayment, loanAmount, paymentType
                )
        return principal_due

    def calculate_dpd(
        df: pd.DataFrame, return_days=False, DUE_DATE=DUE_DATE
    ) -> list:
        """Calculates the days past due for each loan. Used in the parse_dpd function.

        Args:
        ----
        df: loans_proc.
        return_days: setting this to True returns dpd as an integer instead of timedelta.

        Returns:
        ----
        Number of days past due of a loan as timedelta.
        """
        DUE_DATE = DUE_DATE.replace(tzinfo=pytz.utc)
        dpd = []
        for repaidDate, repaymentDate in zip(
            df.loanRepaidDate, df.repaymentDate
        ):
            if pd.isnull(repaidDate) or repaidDate.year == 1970:
                _dpd = DUE_DATE - repaymentDate
                _dpd = _dpd
                if return_days:
                    _dpd = _dpd.days
            else:
                _dpd = repaidDate - repaymentDate
                if return_days:
                    _dpd = _dpd.days
            dpd.append(_dpd)

        return dpd

    def create_category(df: pd.DataFrame, size=180) -> list:
        """Groups the days past due into categories. This will allow us to
        categorise principal collected into <=180 or >180. Used in the parse_dpd
        function.

        Args:
        ----
        df: loans_proc.
        size: category size.

        Returns:
        ----
        Groups loans into categories.
        """

        category = []
        for dpd in df.dpd:
            if 1 <= dpd <= size:
                category.append(f"1-{size}")
            elif dpd <= 0:
                category.append("0")
            elif dpd > size:
                category.append(f">{size}")
            else:
                raise ValueError(f"dpd is unknown: {dpd}")
        return category

    def parse_dpd(self, loans: pd.DataFrame) -> pd.DataFrame:
        """Adds the number of days past due to the loans collection as well as
        categorising the days past due.

        Args:
        ----
        loans: loans_proc.
        """

        loans_dpd = loans.assign(
            dpd=lambda df: self.calculate_dpd(df=df, return_days=True),
            category_180=lambda df: self.create_category(df=df, size=180),
            category_90=lambda df: self.create_category(df=df, size=90),
        )

        return loans_dpd

    def get_loan_collections(
        self, repaymentSchedules: pd.DataFrame
    ) -> pd.DataFrame:
        loan_collections = (
            repaymentSchedules.assign(
                principal_due_=lambda df: self.calculate_principal_due(df),
                principal_due=lambda df: df.principal_due_.astype(float),
            )[["loanId", "totalPaymentWithinSchedule", "principal_due"]]
            .assign(
                totalPaymentWithinSchedule=lambda df: df.totalPaymentWithinSchedule.astype(
                    float
                )
            )
            .groupby("loanId", as_index=False)
            .aggregate(
                amount_collected=("totalPaymentWithinSchedule", "sum"),
                principal_due=("principal_due", "sum"),
            )
        )

        return loan_collections

    def process_loans(
        self, loans: pd.DataFrame, loan_collections: pd.DataFrame
    ) -> pd.DataFrame:
        loans_proc = (
            loans.assign(
                disbursed_ym=lambda df: df.disbursementDateWAT.dt.strftime(
                    "%Y%m"
                ),
                loanRepaidDateWAT=lambda df: df.loanRepaidDate.dt.tz_convert(
                    "Africa/Lagos"
                ),
                loan_due=lambda df: np.where(
                    df.repaymentDateWAT > self.DUE_DATE, True, False
                ),
            )[lambda df: pd.notna(df.disbursed_ym)]
            .merge(right=loan_collections, how="left", on="loanId")
            .assign(
                disbursed_ym=lambda df: df.disbursed_ym.astype(int),
                isRepaid=lambda df: df.status.str.lower()
                .str.strip()
                .str.contains("paid")
                | df.loanRepaidDateWAT.notna(),
                isOutstanding=lambda df: df.amount_collected < df.loanAmount,
            )
            .pipe(self.parse_dpd)
            .query(f"disbursed_ym < {self.EXCLUDE_PERIOD}")
        )

        return loans_proc


if __name__ == "__main__":
    import sklearn

    ap = AirflowPipeline()
    ap.load_data()
    ap.run_train()
    print(sklearn.__version__)
    print(np.__version__)
