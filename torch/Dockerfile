# syntax=docker/dockerfile:1
# FROM freqtradeorg/freqtrade:develop_freqairl
FROM freqtradeorg/freqtrade:develop_freqairl
USER root

RUN apt-get update \
  && apt-get -y install build-essential libssl-dev git libffi-dev libgfortran5 pkg-config cmake gcc

ADD --keep-git-dir=true https://github.com/jersonjunior/freqailstm.git /opt/freqai-lstm
WORKDIR /opt/freqai-lstm

RUN mkdir -p /freqtrade/user_data/strategies /freqtrade/user_data/freqaimodels \
  && cp config/config-tankai.json /freqtrade/user_data/config-torch.json \
  && cp TankAiRevivalFutures.py /freqtrade/user_data/strategies/ \
  && cp torch/PyTorchLSTMRegressor.py /freqtrade/user_data/freqaimodels/ \
  && cp torch/PyTorchModelTrainer.py /freqtrade/user_data/freqaimodels/ \
  && cp torch/BasePyTorchRegressor.py /freqtrade/user_data/freqaimodels/

WORKDIR /freqtrade
RUN sed -i "s/5m/1h/" freqtrade/configuration/config_validation.py
USER ftuser

RUN  pip install -e .

