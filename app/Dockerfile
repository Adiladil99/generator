# syntax=docker/dockerfile:1.4
FROM python:3.11 AS builder

WORKDIR /app

COPY requirements.txt /app
# RUN --mount=type=cache,target=/root/.cache/pip \ 
#     apt-get update && apt-get install -y libgl1-mesa-glx
# RUN --mount=type=cache,target=/root/.cache/pip \ 
#     pip install --upgrade pip
# RUN --mount=type=cache,target=/root/.cache/pip \ 
#     pip3 install numpy opencv-python
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

# ENTRYPOINT ["python -u"]
CMD ["python", "-u", "app.py"]

FROM builder as dev-envs

RUN <<EOF
apk update
apk add git
EOF

RUN <<EOF
addgroup -S docker
adduser -S --shell /bin/bash --ingroup docker vscode
EOF
# install Docker tools (cli, buildx, compose)
COPY --from=gloursdocker/docker / /
