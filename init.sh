#!/bin/sh

# Garante que o arquivo de log existe
if [ ! -f /freqtrade/user_data/logs/freqtrade.log ]; then
  touch /freqtrade/user_data/logs/freqtrade.log
fi

# Aplica permissões somente se possível
chmod 777 /freqtrade/user_data/logs/freqtrade.log || echo "Aviso: não foi possível alterar permissões"

# Executa o comando principal
exec freqtrade "$@"
