FROM python:3.8

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install tensorflow
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
