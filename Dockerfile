FROM nvcr.io/nvidia/tensorflow:18.03-py3

WORKDIR /UppGo
ADD Docker_Requirements.txt ./
RUN pip install -r Docker_Requirements.txt && rm Docker_Requirements.txt
ADD *.py primes1.txt ./
ADD layers ./layers
RUN mkdir logs && mkdir replays && mkdir models

CMD python training.py