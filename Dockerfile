FROM nvcr.io/nvidia/tensorflow:18.03-py3

WORKDIR /UppGo
ADD *.py primes1.txt Docker_Requirements.txt ./
ADD layers ./layers
RUN pip install -r Docker_Requirements.txt && rm Docker_Requirements.txt
RUN mkdir logs && mkdir replays && mkdir models

CMD python training.py