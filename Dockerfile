FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    make \
    libpthread-stubs0-dev \
    netcat \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN make
RUN chmod +x train

