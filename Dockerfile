FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV QT_DEBUG_PLUGINS=1

WORKDIR /GuitarPlay2Tab

RUN apt-get update && pip install --upgrade pip && apt-get install ffmpeg libsm6 libxext6 libegl1 -y
RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

RUN apt install libatomic1

COPY . .

WORKDIR /GuitarPlay2Tab/basic_pitch_torch
RUN pip install .

WORKDIR /GuitarPlay2Tab/lightglue
RUN pip install .

WORKDIR /GuitarPlay2Tab

ENV DISPLAY=host.docker.internal:0.0

CMD ["python", "./play2tab/gui/gui.py"]