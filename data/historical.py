"""
Historical EMR data generator for Healthcare ML Pipeline.
Generates realistic synthetic patient data with correlated features.
"""

import logging
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

DIAGNOSIS_CODES = [
    "I10",  # Hypertension
    "E11",  # Type 2 Diabetes
    "J44",  # COPD
    "I25",  # Chronic Ischemic Heart Disease
    "N18",  # Chronic Kidney Disease
    "I50",  # Heart Failure
    "J18",  # Pneumonia
    "A41",  # Sepsis
    "K92",  # GI Hemorrhage
    "I63",  # Stroke
]

ADMISSION_TYPES = ["Emergency", "Urgent", "Elective"]
INSURANCE_TYPES = ["Private", "Medicare", "Medicaid", "Uninsured"]
GENDERS = ["M", "F", "Other"]


class HistoricalDataGenerator:
    def __init__(self, session: Session, database: str, schema_name: str):
        self.session = session
        self.database = database
        self.schema_name = schema_name
        self.full_schema = f"{database}.{schema_name}"

    def _generate_patient_id(self) -> str:
        return f"P{uuid.uuid4().hex[:8].upper()}"

    def _generate_encounter_id(self) -> str:
        return f"E{uuid.uuid4().hex[:10].upper()}"

    def _generate_vital_signs(self, age: int, risk_factor: float) -> dict:
        base_hr = 75 + (age - 50) * 0.2
        base_sbp = 120 + (age - 40) * 0.5
        base_dbp = 80 + (age - 40) * 0.3

        hr_adj = base_hr + risk_factor * 25 + np.random.normal(0, 3)
        sbp_adj = base_sbp + risk_factor * 35 + np.random.normal(0, 5)
        dbp_adj = base_dbp + risk_factor * 15 + np.random.normal(0, 3)

        temp = 36.8 + risk_factor * 1.2 + np.random.normal(0, 0.15)
        rr = 16 + risk_factor * 8 + np.random.normal(0, 1)
        spo2 = 98 - risk_factor * 8 + np.random.normal(0, 0.5)

        return {
            "heart_rate": int(np.clip(hr_adj, 40, 180)),
            "systolic_bp": int(np.clip(sbp_adj, 70, 220)),
            "diastolic_bp": int(np.clip(dbp_adj, 40, 130)),
            "temperature": round(np.clip(temp, 35.5, 41.0), 1),
            "respiratory_rate": int(np.clip(rr, 8, 40)),
            "oxygen_saturation": round(np.clip(spo2, 70, 100), 1),
        }

    def _generate_lab_values(
        self, age: int, risk_factor: float
    ) -> dict:  # pylint: disable=unused-argument
        glucose = 100 + risk_factor * 120 + np.random.normal(0, 8)
        creatinine = 1.0 + risk_factor * 2.5 + np.random.normal(0, 0.1)
        hemoglobin = 14 - risk_factor * 5 + np.random.normal(0, 0.5)
        wbc = 7.5 + risk_factor * 10 + np.random.normal(0, 1)

        return {
            "glucose_level": round(np.clip(glucose, 40, 500), 1),
            "creatinine": round(np.clip(creatinine, 0.3, 10.0), 2),
            "hemoglobin": round(np.clip(hemoglobin, 5, 18), 1),
            "wbc_count": round(np.clip(wbc, 1, 30), 1),
        }

    def _calculate_risk_level(
        self,
        age: int,
        vital_signs: dict,
        lab_values: dict,
        comorbidities: int,
        previous_admissions: int,
    ) -> str:
        score = 0

        if age >= 75:
            score += 2
        elif age >= 65:
            score += 1

        if vital_signs["heart_rate"] > 100 or vital_signs["heart_rate"] < 50:
            score += 1
        if vital_signs["systolic_bp"] > 180 or vital_signs["systolic_bp"] < 90:
            score += 2
        if vital_signs["temperature"] > 38.5 or vital_signs["temperature"] < 36.0:
            score += 1
        if vital_signs["oxygen_saturation"] < 92:
            score += 2
        elif vital_signs["oxygen_saturation"] < 95:
            score += 1
        if vital_signs["respiratory_rate"] > 24:
            score += 1

        if lab_values["glucose_level"] > 200:
            score += 1
        if lab_values["creatinine"] > 2.0:
            score += 2
        elif lab_values["creatinine"] > 1.5:
            score += 1
        if lab_values["hemoglobin"] < 10:
            score += 1
        if lab_values["wbc_count"] > 12 or lab_values["wbc_count"] < 4:
            score += 1

        if comorbidities >= 4:
            score += 2
        elif comorbidities >= 2:
            score += 1

        if previous_admissions >= 3:
            score += 2
        elif previous_admissions >= 1:
            score += 1

        if score <= 2:
            return "LOW"
        elif score <= 5:
            return "MEDIUM"
        elif score <= 8:
            return "HIGH"
        else:
            return "CRITICAL"

    def generate_patient_data(self, num_records: int) -> pd.DataFrame:
        logger.info(f"Generating {num_records} patient records")

        records = []
        base_time = datetime.now() - timedelta(days=365)

        for i in range(num_records):
            age = int(np.clip(np.random.normal(60, 15), 18, 100))
            gender = np.random.choice(GENDERS, p=[0.48, 0.48, 0.04])

            base_bmi = 26 if gender == "M" else 25
            bmi = round(np.clip(np.random.normal(base_bmi, 5), 16, 50), 1)

            target_risk = np.random.choice(
                [0.0, 0.33, 0.66, 1.0], p=[0.40, 0.35, 0.18, 0.07]
            )
            risk_factor = np.clip(target_risk + np.random.normal(0, 0.08), 0, 1)

            vital_signs = self._generate_vital_signs(age, risk_factor)
            lab_values = self._generate_lab_values(age, risk_factor)

            comorbidities = int(np.clip(np.random.poisson(2 + risk_factor * 3), 0, 10))
            previous_admissions = int(
                np.clip(np.random.poisson(1 + risk_factor * 2), 0, 10)
            )
            medication_count = int(
                np.clip(np.random.poisson(5 + comorbidities * 2), 0, 20)
            )

            diagnosis = np.random.choice(DIAGNOSIS_CODES)
            admission_type = np.random.choice(
                ADMISSION_TYPES,
                p=[0.4 + risk_factor * 0.2, 0.35, 0.25 - risk_factor * 0.2],
            )
            insurance_type = np.random.choice(
                INSURANCE_TYPES,
                p=[0.35, 0.35, 0.20, 0.10] if age >= 65 else [0.50, 0.15, 0.25, 0.10],
            )

            risk_level = self._calculate_risk_level(
                age=age,
                vital_signs=vital_signs,
                lab_values=lab_values,
                comorbidities=comorbidities,
                previous_admissions=previous_admissions,
            )

            timestamp = base_time + timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60),
            )

            record = {
                "PATIENT_ID": self._generate_patient_id(),
                "ENCOUNTER_ID": self._generate_encounter_id(),
                "TIMESTAMP": timestamp,
                "AGE": age,
                "GENDER": gender,
                "BMI": bmi,
                "HEART_RATE": vital_signs["heart_rate"],
                "SYSTOLIC_BP": vital_signs["systolic_bp"],
                "DIASTOLIC_BP": vital_signs["diastolic_bp"],
                "TEMPERATURE": vital_signs["temperature"],
                "RESPIRATORY_RATE": vital_signs["respiratory_rate"],
                "OXYGEN_SATURATION": vital_signs["oxygen_saturation"],
                "GLUCOSE_LEVEL": lab_values["glucose_level"],
                "CREATININE": lab_values["creatinine"],
                "HEMOGLOBIN": lab_values["hemoglobin"],
                "WBC_COUNT": lab_values["wbc_count"],
                "PRIMARY_DIAGNOSIS": diagnosis,
                "COMORBIDITY_COUNT": comorbidities,
                "ADMISSION_TYPE": admission_type,
                "INSURANCE_TYPE": insurance_type,
                "PREVIOUS_ADMISSIONS": previous_admissions,
                "MEDICATION_COUNT": medication_count,
                "RISK_LEVEL": risk_level,
            }
            records.append(record)

            if (i + 1) % 10000 == 0:
                logger.info(f"Generated {i + 1}/{num_records} records")

        df = pd.DataFrame(records)

        risk_dist = df["RISK_LEVEL"].value_counts(normalize=True)
        logger.info(f"Risk level distribution:\n{risk_dist}")

        return df

    def load_to_snowflake(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str = "overwrite",
    ) -> dict:
        full_table = f"{self.full_schema}.{table_name}"
        logger.info(f"Loading {len(df)} records to {full_table}")

        snowpark_df = self.session.create_dataframe(df)
        snowpark_df.write.mode(mode).save_as_table(full_table)

        count = self.session.table(full_table).count()
        logger.info(f"Loaded {count} records to {full_table}")

        return {"table": full_table, "records": count}

    def create_baseline_sample(
        self,
        source_table: str,
        baseline_table: str,
        sample_fraction: float = 0.2,
    ) -> dict:
        source = f"{self.full_schema}.{source_table}"
        target = f"{self.full_schema}.{baseline_table}"

        logger.info(f"Creating baseline sample from {source} to {target}")

        self.session.sql(
            f"""
            CREATE OR REPLACE TABLE {target} AS
            SELECT * FROM {source}
            SAMPLE ({sample_fraction * 100})
        """
        ).collect()

        count = self.session.table(target).count()
        logger.info(f"Created baseline with {count} records")

        return {"table": target, "records": count}

    def create_test_split(
        self,
        source_table: str,
        test_table: str,
        test_fraction: float = 0.2,
    ) -> dict:
        source = f"{self.full_schema}.{source_table}"
        target = f"{self.full_schema}.{test_table}"

        logger.info(f"Creating test split from {source} to {target}")

        self.session.sql(
            f"""
            CREATE OR REPLACE TABLE {target} AS
            SELECT * FROM {source}
            SAMPLE ({test_fraction * 100})
        """
        ).collect()

        count = self.session.table(target).count()
        logger.info(f"Created test set with {count} records")

        return {"table": target, "records": count}

    def run(
        self,
        num_records: int = 50000,
        table_name: str = "RAW_PATIENT_DATA",
        create_baseline: bool = True,
        create_test_split: bool = True,
    ) -> dict:
        logger.info(f"Running historical data generation: {num_records} records")

        df = self.generate_patient_data(num_records)
        load_result = self.load_to_snowflake(df, table_name)

        results = {
            "main_table": load_result,
        }

        if create_baseline:
            baseline_result = self.create_baseline_sample(
                source_table=table_name,
                baseline_table="BASELINE_PATIENT_DATA",
            )
            results["baseline_table"] = baseline_result

        if create_test_split:
            test_result = self.create_test_split(
                source_table=table_name,
                test_table="TEST_PATIENT_DATA",
            )
            results["test_table"] = test_result

        logger.info("Historical data generation complete")
        results["status"] = "success"

        return results


def main():
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from configs import get_config
    from source.utils import get_session

    logging.basicConfig(level=logging.INFO)

    config = get_config()
    session = get_session(config.snowflake.connection_name)
    session.use_database(config.snowflake.database)
    session.use_schema(config.snowflake.schema_name)
    session.use_warehouse(config.snowflake.warehouse)

    generator = HistoricalDataGenerator(
        session=session,
        database=config.snowflake.database,
        schema_name=config.snowflake.schema_name,
    )

    result = generator.run(num_records=50000)
    logger.info(f"Result: {result}")
    return result


if __name__ == "__main__":
    main()
