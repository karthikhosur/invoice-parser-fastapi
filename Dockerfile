FROM python:3.9

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN  apt-get update && sudo apt-get  -y  install ffmpeg

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./main.py /code/

# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
