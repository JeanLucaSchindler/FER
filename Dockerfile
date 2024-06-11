FROM python:3.10.6

#WORKDIR /prod

# First, pip install dependencies
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

# Then only, install taxifare!
COPY FER_directory /FER_directory
COPY setup.py /setup.py
RUN pip install .


CMD uvicorn fast:app --reload
