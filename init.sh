#!/bin/sh
chmod 777 /freqtrade/user_data/logs/freqtrade.log
exec "$@"
