# syntax=docker/dockerfile:1
# FROM freqtradeorg/freqtrade:develop_freqairl
FROM freqtradeorg/freqtrade:develop_freqairl
USER root

RUN apt-get update \
  && apt-get -y install build-essential libssl-dev git libffi-dev libgfortran5 pkg-config cmake gcc

ADD --keep-git-dir=true https://github.com/jersonjunior/freqailstm.git /opt/freqai-lstm
WORKDIR /opt/freqai-lstm

RUN mkdir -p /freqtrade/user_data/strategies /freqtrade/user_data/freqaimodels \
  && cp user_data/config-tankai.json /freqtrade/user_data/config-torch.json \
  && cp user_data/strategies/TankAiRevivalFuturesNew.py /freqtrade/user_data/strategies/ \
  && cp torch/PyTorchDataConvertor.py /freqtrade/user_data/freqaimodels/ \
  && cp torch/PyTorchLSTMModel.py /freqtrade/user_data/freqaimodels/ \
  && cp torch/PyTorchLSTMModelTrainer.py /freqtrade/user_data/freqaimodels/ \
  && cp torch/PyTorchLSTMTrainer.py /freqtrade/user_data/freqaimodels/ \
  && cp torch/PyTorchLSTMTransformerTrainer.py /freqtrade/user_data/freqaimodels/ \
  && cp torch/PyTorchTrainerInterface.py /freqtrade/user_data/freqaimodels/ \
  && cp torch/datasets.py /freqtrade/user_data/freqaimodels/

WORKDIR /freqtrade
#RUN sed -i "s/5m/1h/" freqtrade/configuration/config_validation.py
USER ftuser

RUN  pip install -e .
