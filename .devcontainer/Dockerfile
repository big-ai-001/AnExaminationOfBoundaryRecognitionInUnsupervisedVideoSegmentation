FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3

COPY ./requirements.txt ./requirements.txt
COPY ./init.py ./init.py

RUN python -m pip install --upgrade pip \
 && pip install pip setuptools wheel -U \
 && pip install -r ./requirements.txt \
 && python ./init.py \
 && python -m spacy download en_core_web_trf \
 && mkdir -p /workspaces/AnExaminationOfBoundaryRecognitionInUnsupervisedVideoSegmentation/model \
 && cd /workspaces/AnExaminationOfBoundaryRecognitionInUnsupervisedVideoSegmentation/model \
 && wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip \
 && unzip uncased_L-12_H-768_A-12.zip

# RUN python -m pip install --upgrade pip \
#  && pip install pip setuptools wheel -U \
#  && pip install -r ./requirements.txt \
#  && python ./init.py \
#  && python -m spacy download en_core_web_trf \

# RUN /usr/bin/python -m pip install --upgrade pip \
#  && pip install pip setuptools wheel -U \
#  && cd /workspaces/AnExaminationOfBoundaryRecognitionInUnsupervisedVideoSegmentation \
#  && pip install -r ./code/requirements.txt \
#  && python ./code/init.py \
#  && python -m spacy download en_core_web_trf \
#  && cd ./code/model \
#  && wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip \
#  && unzip uncased_L-12_H-768_A-12.zip