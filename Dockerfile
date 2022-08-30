FROM python:3.8.13-bullseye

COPY fakenews /fakenews
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt -d /usr/local/nltk_data
RUN python -m nltk.downloader stopwords -d /usr/local/nltk_data
RUN python -m nltk.downloader wordnet -d /usr/local/nltk_data
RUN python -m nltk.downloader omw-1.4 -d /usr/local/nltk_data


CMD uvicorn fakenews.api.fast:app --host 0.0.0.0 --port $PORT
