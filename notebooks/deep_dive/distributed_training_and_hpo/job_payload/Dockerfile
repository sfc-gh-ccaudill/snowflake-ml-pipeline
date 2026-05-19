ARG SNOWFLAKE_REGISTRY=<DEFAULT_VALUE>
FROM ${SNOWFLAKE_REGISTRY}/snowflake/images/snowflake_images/st_plat/runtime/x86/runtime_image/snowbooks:2.5.1-py311

RUN uv pip install --system --break-system-packages --no-cache \
    scikit-learn \
    xgboost
