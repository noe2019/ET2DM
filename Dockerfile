FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app

<<<<<<< HEAD
EXPOSE 8002

CMD [ "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8002"]
=======
EXPOSE 8003

CMD [ "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8003"]
>>>>>>> 2406caacd078552d864f1b331d73bd4512154235
