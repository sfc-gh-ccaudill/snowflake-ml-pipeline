import json
import os

from dotenv import load_dotenv
import requests

load_dotenv()
REST_URL = "<REPLACE_WITH_REST_URL>"
ENDPOINT_URL = f"https://{REST_URL}/predict"
SNOWFLAKE_TOKEN = os.getenv("SNOWFLAKE_TOKEN")

patient_data = {
    "AGE": 65,
    "BMI": 28.5,
    "HEART_RATE": 88,
    "SYSTOLIC_BP": 140,
    "DIASTOLIC_BP": 85,
    "TEMPERATURE": 37.2,
    "RESPIRATORY_RATE": 18,
    "OXYGEN_SATURATION": 96,
    "GLUCOSE_LEVEL": 110,
    "CREATININE": 1.1,
    "HEMOGLOBIN": 13.5,
    "WBC_COUNT": 7.2,
    "COMORBIDITY_COUNT": 3,
    "PREVIOUS_ADMISSIONS": 2,
    "MEDICATION_COUNT": 5,
    "GENDER": "M",
    "PRIMARY_DIAGNOSIS": "CARDIAC",
    "ADMISSION_TYPE": "EMERGENCY",
    "INSURANCE_TYPE": "MEDICARE",
    "SHOCK_INDEX": 88 / 140,
    "PULSE_PRESSURE": 140 - 85,
    "BMI_CATEGORY": "OVERWEIGHT",
    "VITAL_SIGNS_SEVERITY": 1,
}

payload = {"dataframe_records": [patient_data]}

headers = {
    "Content-Type": "application/json",
    "Authorization": f'Snowflake Token="{SNOWFLAKE_TOKEN}"',
}

print("Input features:")
for k, v in patient_data.items():
    print(f"  {k}: {v}")
print()

response = requests.post(ENDPOINT_URL, headers=headers, json=payload)

print(f"Status: {response.status_code}")
try:
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except requests.exceptions.JSONDecodeError:
    print(f"Response: {response.text}")
