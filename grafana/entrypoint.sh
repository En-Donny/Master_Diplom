#!/bin/sh
set -e

if [ -f /run/secrets/influxdb2_admin_token ]; then
  export INFLUXDB_TOKEN="$(cat /run/secrets/influxdb2_admin_token)"
fi

exec /run.sh