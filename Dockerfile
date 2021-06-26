# pull official base image
FROM python:3.8-slim-buster

# set work directory
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
# RUN pip install --upgrade pip && \
#     apt-get update && \
#     apt-get install -y python3-dev libpq-dev gcc
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# copy project
COPY . ./app/