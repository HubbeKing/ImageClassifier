FROM python:2

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

# By default, run main.py -h
# By laying it out as ENTRYPOINT and CMD, one can just do "docker run IMAGE train /path/to/data" to train the model
ENTRYPOINT ["python", "main.py"]
CMD ["-h"]

# Tensorboard hosts on port 6006 by default, so expose it
EXPOSE 6006
