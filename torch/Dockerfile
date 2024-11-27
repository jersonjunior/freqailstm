# syntax=docker/dockerfile:1
# FROM freqtradeorg/freqtrade:develop_freqairl
FROM freqtradeorg/freqtrade:develop_freqairl
USER root

RUN apt-get update \
  && apt-get -y install build-essential libssl-dev git libffi-dev libgfortran5 pkg-config cmake gcc

ADD --keep-git-dir=true https://github.com/jersonjunior/freqailstm.git /opt/freqai-lstm
WORKDIR /opt/freqai-lstm

RUN mkdir -p /freqtrade/user_data/strategies /freqtrade/user_data/freqaimodels \
  && cp config/config-torch.json /freqtrade/user_data/config-torch.json \
  && cp config/config_ai.json /freqtrade/user_data/config_ai.json \
  && cp config/config-TankAiRevival-PC.json /freqtrade/user_data/config-TankAiRevival-PC.json
  && cp config/config_backtesting_notankai.json /freqtrade/user_data/config_backtesting_notankai.json \
  && cp V9/AlexBTStrategyFinalV93.py /freqtrade/user_data/strategies/ \
  && cp V9/TankAiRevival.py /freqtrade/user_data/strategies/ \
  && cp V9/NOTankAi_15.py /freqtrade/user_data/strategies/ \
  #&& cp V9/TGDF.py /freqtrade/user_data/strategies/ \
  && cp torch/BasePyTorchModel.py /freqtrade/freqtrade/freqai/base_models/ \
  && cp torch/PyTorchLSTMModel.py /freqtrade/freqtrade/freqai/torch/ \
  && cp torch/PyTorchModelTrainer.py /freqtrade/freqtrade/freqai/torch/ \
  && cp torch/PyTorchLSTMRegressor.py /freqtrade/user_data/freqaimodels/

WORKDIR /freqtrade
RUN sed -i "s/5m/1h/" freqtrade/configuration/config_validation.py
USER ftuser

RUN  pip install -e . 
