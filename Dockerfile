FROM python:2.7

WORKDIR /usr/app

COPY requirements.txt .

RUN pip install -r requirements.txt 

COPY . .

ENTRYPOINT [ "python" ]