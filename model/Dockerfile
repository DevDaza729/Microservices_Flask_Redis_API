FROM python:3.8.13 AS base

ARG UID=501
ARG GID=570

# Add a non-root user
RUN addgroup --gid $GID app
RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app
USER app

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=/src/
ENV PATH=$PATH:/home/app/.local/bin/

ADD requirements.txt .
# Option is used to cache the pip packages
# RUN --mount=type=cache,target=/root/.cache/pip pip3 install --user --upgrade pip
RUN pip3 install -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/src/

COPY ./ /src/

WORKDIR /src

FROM base AS test
RUN ["pytest", "-v", "/src/tests"]

FROM base AS build
ENTRYPOINT ["python3", "/src/ml_service.py"]

