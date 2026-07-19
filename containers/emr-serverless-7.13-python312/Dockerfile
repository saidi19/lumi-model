FROM public.ecr.aws/emr-serverless/spark/emr-7.13.0@sha256:3e465888c2b00add9d298a2b4c14c7951262d9c2b35a3f8c1b109dc21cd9714d

USER root

COPY requirements-linux-x86_64.lock /opt/lumi/requirements-linux-x86_64.lock
COPY verify_runtime.py /opt/lumi/verify_runtime.py

RUN yum install -y python3.12 python3.12-pip \
    && yum clean all \
    && rm -rf /var/cache/yum \
    && /usr/bin/python3.12 -m venv /opt/lumi/python312 \
    && /opt/lumi/python312/bin/python -m pip install \
        --no-cache-dir \
        --only-binary=:all: \
        --require-hashes \
        --requirement /opt/lumi/requirements-linux-x86_64.lock \
    && /opt/lumi/python312/bin/python /opt/lumi/verify_runtime.py \
    && chown -R hadoop:hadoop /opt/lumi

# EMR Serverless requires the final image user to remain hadoop:hadoop.
# JAVA_HOME, SPARK_HOME, HIVE_HOME, and TEZ_HOME intentionally remain untouched.
USER hadoop:hadoop
