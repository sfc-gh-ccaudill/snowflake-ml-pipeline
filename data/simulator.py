"""
Streaming data simulator for Healthcare ML Pipeline.
Continuously generates patient data at specified intervals.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

DIAGNOSIS_CODES = [
    "I10",
    "E11",
    "J44",
    "I25",
    "N18",
    "I50",
    "J18",
    "A41",
    "K92",
    "I63",
]

ADMISSION_TYPES = ["Emergency", "Urgent", "Elective"]
INSURANCE_TYPES = ["Private", "Medicare", "Medicaid", "Uninsured"]
GENDERS = ["M", "F", "Other"]

DRIFT_TYPES = {
    "age_shift": {"column": "AGE", "shift": 10},
    "vital_degradation": {
        "columns": ["HEART_RATE", "SYSTOLIC_BP", "OXYGEN_SATURATION"]
    },
    "feature_scale": {"columns": ["GLUCOSE_LEVEL", "CREATININE"], "factor": 1.3},
    "distribution_shift": {"column": "ADMISSION_TYPE", "new_probs": [0.7, 0.2, 0.1]},
}


class StreamingDataSimulator:
    def __init__(self, session: Session, database: str, schema_name: str):
        self.session = session
        self.database = database
        self.schema_name = schema_name
        self.full_schema = f"{database}.{schema_name}"
        self.records_generated = 0
        self.drift_enabled = False
        self.drift_type = None

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

    def generate_streaming_record(self) -> dict:
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
        medication_count = int(np.clip(np.random.poisson(5 + comorbidities * 2), 0, 20))

        diagnosis = np.random.choice(DIAGNOSIS_CODES)
        admission_type = np.random.choice(
            ADMISSION_TYPES, p=[0.4 + risk_factor * 0.2, 0.35, 0.25 - risk_factor * 0.2]
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

        record = {
            "PATIENT_ID": self._generate_patient_id(),
            "ENCOUNTER_ID": self._generate_encounter_id(),
            "TIMESTAMP": datetime.now(),
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

        return record

    def introduce_drift(self, record: dict, drift_type: str) -> dict:
        if drift_type not in DRIFT_TYPES:
            logger.warning(f"Unknown drift type: {drift_type}")
            return record

        drift_config = DRIFT_TYPES[drift_type]

        if drift_type == "age_shift":
            record["AGE"] = min(100, record["AGE"] + drift_config["shift"])

        elif drift_type == "vital_degradation":
            record["HEART_RATE"] = min(180, int(record["HEART_RATE"] * 1.15))
            record["SYSTOLIC_BP"] = min(220, int(record["SYSTOLIC_BP"] * 1.1))
            record["OXYGEN_SATURATION"] = max(70, record["OXYGEN_SATURATION"] - 3)

        elif drift_type == "feature_scale":
            factor = drift_config["factor"]
            record["GLUCOSE_LEVEL"] = round(record["GLUCOSE_LEVEL"] * factor, 1)
            record["CREATININE"] = round(record["CREATININE"] * factor, 2)

        elif drift_type == "distribution_shift":
            record["ADMISSION_TYPE"] = np.random.choice(
                ADMISSION_TYPES, p=drift_config["new_probs"]
            )

        return record

    def enable_drift(self, drift_type: str) -> None:
        if drift_type in DRIFT_TYPES:
            self.drift_enabled = True
            self.drift_type = drift_type
            logger.info(f"Drift enabled: {drift_type}")
        else:
            logger.warning(f"Unknown drift type: {drift_type}")

    def disable_drift(self) -> None:
        self.drift_enabled = False
        self.drift_type = None
        logger.info("Drift disabled")

    def insert_record(self, record: dict, table_name: str) -> None:
        full_table = f"{self.full_schema}.{table_name}"
        df = pd.DataFrame([record])
        snowpark_df = self.session.create_dataframe(df)
        snowpark_df.write.mode("append").save_as_table(full_table, column_order="name")

    def insert_batch(self, records: list, table_name: str) -> None:
        if not records:
            return

        full_table = f"{self.full_schema}.{table_name}"
        df = pd.DataFrame(records)
        snowpark_df = self.session.create_dataframe(df)
        snowpark_df.write.mode("append").save_as_table(full_table, column_order="name")

    def run(
        self,
        interval_seconds: float = 1.0,
        duration_minutes: int = 60,
        batch_size: int = 10,
        table_name: str = "STREAMING_PATIENT_DATA",
        enable_drift_after: Optional[int] = None,
        drift_type: Optional[str] = None,
    ) -> dict:
        logger.info(f"Starting streaming simulator")
        logger.info(f"  Interval: {interval_seconds}s, Duration: {duration_minutes}min")
        logger.info(f"  Batch size: {batch_size}, Table: {table_name}")

        if enable_drift_after and drift_type:
            logger.info(f"  Drift will be enabled after {enable_drift_after} records")

        total_iterations = int((duration_minutes * 60) / interval_seconds)
        start_time = datetime.now()
        batch = []

        try:
            for _ in range(total_iterations):
                record = self.generate_streaming_record()

                if enable_drift_after and self.records_generated >= enable_drift_after:
                    if not self.drift_enabled and drift_type:
                        self.enable_drift(drift_type)

                if self.drift_enabled and self.drift_type:
                    record = self.introduce_drift(record, self.drift_type)

                batch.append(record)
                self.records_generated += 1

                if len(batch) >= batch_size:
                    self.insert_batch(batch, table_name)
                    batch = []
                    logger.info(
                        f"Inserted batch. Total records: {self.records_generated}"
                    )

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")

        finally:
            if batch:
                self.insert_batch(batch, table_name)
                logger.info(f"Inserted final batch of {len(batch)} records")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = {
            "records_generated": self.records_generated,
            "duration_seconds": duration,
            "records_per_second": (
                self.records_generated / duration if duration > 0 else 0
            ),
            "drift_enabled": self.drift_enabled,
            "drift_type": self.drift_type,
            "status": "success",
        }

        logger.info(f"Streaming complete: {result}")
        return result


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

    simulator = StreamingDataSimulator(
        session=session,
        database=config.snowflake.database,
        schema_name=config.snowflake.schema_name,
    )

    result = simulator.run(
        interval_seconds=1.0,
        duration_minutes=5,
        batch_size=10,
        enable_drift_after=100,
        drift_type="vital_degradation",
    )

    logger.info(f"Result: {result}")
    return result


if __name__ == "__main__":
    main()
