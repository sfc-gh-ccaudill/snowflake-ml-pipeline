-- This can be used without a secret for public github repos suchn as following my demos
create or replace api integration git_int
    api_provider = git_https_api
    api_allowed_prefixes = ('https://github.com')
    enabled = true
    allowed_authentication_secrets = all;

-- Only needed for private repos or to make changes
CREATE OR REPLACE SECRET YOUR_SECRET
  TYPE = password
  USERNAME = ''
  PASSWORD = ''; -- Put your secret here

-- Needed to pip install in notebooks
CREATE OR REPLACE NETWORK RULE pypi_network_rule
MODE = EGRESS
TYPE = HOST_PORT
VALUE_LIST = ('pypi.org','raw.githubusercontent.com', 'pypi.python.org', 'pythonhosted.org',  'files.pythonhosted.org');

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION pypi_access_integration
ALLOWED_NETWORK_RULES = (pypi_network_rule)
ENABLED = true;

-- replace sysadmin with your data scientists role
GRANT USAGE ON INTEGRATION pypi_access_integration TO ROLE sysadmin;
GRANT USAGE ON INTEGRATION git_int TO ROLE sysadmin; 

-- GPU Compute Pool
CREATE COMPUTE POOL notebook_GPU_S
MIN_NODES = 1
MAX_NODES = 1
INSTANCE_FAMILY = GPU_NV_S;

-- CPU Compute Pool
CREATE COMPUTE POOL notebook_CPU_S
  MIN_NODES = 1
  MAX_NODES = 4
  INSTANCE_FAMILY = CPU_X64_S;

grant usage on compute pool notebook_GPU_S to role sysadmin;
grant usage on compute pool notebook_CPU_S to role sysadmin;