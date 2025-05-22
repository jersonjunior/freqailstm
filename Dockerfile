# syntax=docker/dockerfile:1
#FROM freqtradeorg/freqtrade:develop_freqairl
FROM freqtradeorg/freqtrade:develop
USER root

RUN apt-get update \
  && apt-get -y install build-essential libssl-dev git libffi-dev libgfortran5 pkg-config cmake gcc

# Install additional Python dependencies via pip
RUN pip install --no-cache-dir optuna \
  && pip install --no-cache-dir torch \
  && pip install --no-cache-dir sb3_contrib \
  && pip install --no-cache-dir datasieve

ADD --keep-git-dir=true https://github.com/jersonjunior/freqailstm.git /opt/freqai-lstm
WORKDIR /opt/freqai-lstm

RUN mkdir -p /freqtrade/user_data/strategies /freqtrade/user_data/freqaimodels \
  #&& cp user_data/config-tankai.json /freqtrade/user_data/config-torch.json \
  #&& cp user_data/strategies/NOTankAi152F.py /freqtrade/user_data/strategies/ \
  #
  && cp torch/PyTorchDataConvertor.py /freqtrade/freqtrade/freqai/torch/ \
  && cp torch/PyTorchLSTMModel.py /freqtrade/freqtrade/freqai/torch/ \
  && cp torch/PyTorchLSTMModelTrainer.py /freqtrade/freqtrade/freqai/torch/ \
  && cp torch/PyTorchLSTMTrainer.py /freqtrade/freqtrade/freqai/torch/ \
  && cp torch/PyTorchLSTMTransformerTrainer.py /freqtrade/freqtrade/freqai/torch/ \
  && cp torch/PyTorchTrainerInterface.py /freqtrade/freqtrade/freqai/torch/ \
  #&& cp torch/datasets.py /freqtrade/freqtrade/user_data/freqaimodels/ \
  #&& cp torch/PyTorchLSTMRegressor.py /freqtrade/user_data/freqaimodels/ \
  && cp torch/BasePyLSTMTorchModel.py /freqtrade/freqtrade/freqai/base_models/ \
  && cp torch/BasePyLSTMTorchRegressor.py /freqtrade/freqtrade/freqai/base_models/ 
  #&& cp torch/PyTorchLSTMTransformerRegressor.py /freqtrade/user_data/freqaimodels/ 
  
WORKDIR /freqtrade
#RUN sed -i "s/5m/1h/" freqtrade/configuration/config_validation.py
USER ftuser

RUN  pip install -e .
