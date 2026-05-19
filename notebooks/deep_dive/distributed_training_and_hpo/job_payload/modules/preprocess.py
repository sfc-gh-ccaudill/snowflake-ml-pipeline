"""
Shared preprocessing module for all training entrypoints.

Centralizes column definitions, feature engineering, and target encoding
so that individual training scripts can focus on their primary purpose.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "AGE",
    "BMI",
    "HEART_RATE",
    "SYSTOLIC_BP",
    "DIASTOLIC_BP",
    "TEMPERATURE",
    "RESPIRATORY_RATE",
    "OXYGEN_SATURATION",
    "GLUCOSE_LEVEL",
    "CREATININE",
    "HEMOGLOBIN",
    "WBC_COUNT",
    "COMORBIDITY_COUNT",
    "PREVIOUS_ADMISSIONS",
    "MEDICATION_COUNT",
]
CATEGORICAL_COLS = ["GENDER", "PRIMARY_DIAGNOSIS", "ADMISSION_TYPE", "INSURANCE_TYPE"]
TARGET_COL = "RISK_LEVEL"
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


class Preprocessor:
    """Builds and applies a sklearn ColumnTransformer + LabelEncoder."""

    def __init__(self, numeric_cols=None, categorical_cols=None, target_col=None):
        self.numeric_cols = numeric_cols or NUMERIC_COLS
        self.categorical_cols = categorical_cols or CATEGORICAL_COLS
        self.target_col = target_col or TARGET_COL
        self.feature_cols = self.numeric_cols + self.categorical_cols
        self.column_transformer = None
        self.label_encoder = None

    def build_column_transformer(self):
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        self.column_transformer = ColumnTransformer(
            [
                ("num", numeric_pipe, self.numeric_cols),
                ("cat", categorical_pipe, self.categorical_cols),
            ],
            remainder="drop",
        )
        return self.column_transformer

    def fit_transform(self, train_df, test_df=None):
        if self.column_transformer is None:
            self.build_column_transformer()

        X_train = self.column_transformer.fit_transform(train_df[self.feature_cols])
        X_test = (
            self.column_transformer.transform(test_df[self.feature_cols])
            if test_df is not None
            else None
        )

        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_df[self.target_col])
        y_test = (
            self.label_encoder.transform(test_df[self.target_col]) if test_df is not None else None
        )

        return X_train, y_train, X_test, y_test

    def sample_input(self, df, n=10):
        sample = df[self.feature_cols].head(n).copy()
        for col in sample.select_dtypes(include=["object"]).columns:
            sample[col] = sample[col].fillna("Unknown")
        sample = sample.fillna(0)
        return sample
