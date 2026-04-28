#!/bin/sh
# Grafana auto-setup — creates Prometheus datasource + inference dashboard
# Runs as a sidecar; safe to re-run (idempotent)

# Install curl (not in alpine by default)
apk add --no-cache curl > /dev/null 2>&1

GRAFANA="http://grafana:3000"
AUTH="admin:admin"

echo "[grafana-setup] Waiting for Grafana..."
i=0
until curl -sf "$GRAFANA/api/health" > /dev/null 2>&1; do
  i=$((i+1))
  if [ $i -gt 40 ]; then echo "[grafana-setup] Timeout"; exit 1; fi
  sleep 3
done
echo "[grafana-setup] Grafana ready"

# Datasource
DS_EXISTS=$(curl -s -u "$AUTH" "$GRAFANA/api/datasources/name/Prometheus" | grep -c '"id"' 2>/dev/null || echo 0)
if [ "$DS_EXISTS" = "0" ]; then
  curl -sf -u "$AUTH" -X POST "$GRAFANA/api/datasources" \
    -H "Content-Type: application/json" \
    -d '{"name":"Prometheus","type":"prometheus","url":"http://prometheus:9090","access":"proxy","isDefault":true}' > /dev/null
  echo "[grafana-setup] Datasource created"
else
  echo "[grafana-setup] Datasource already exists"
fi

DS_UID=$(curl -s -u "$AUTH" "$GRAFANA/api/datasources/name/Prometheus" | \
  grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
echo "[grafana-setup] DS UID: $DS_UID"

curl -sf -u "$AUTH" -X POST "$GRAFANA/api/dashboards/import" \
  -H "Content-Type: application/json" \
  -d "{
    \"overwrite\": true,
    \"inputs\": [{\"name\":\"DS_PROMETHEUS\",\"type\":\"datasource\",\"pluginId\":\"prometheus\",\"value\":\"$DS_UID\"}],
    \"dashboard\": {
      \"title\": \"Medical Imaging — Inference Dashboard\",
      \"uid\": \"medical-imaging-v1\",
      \"tags\": [\"medical-imaging\",\"mlops\"],
      \"refresh\": \"10s\",
      \"time\": {\"from\":\"now-1h\",\"to\":\"now\"},
      \"schemaVersion\": 38,
      \"panels\": [
        {\"id\":1,\"gridPos\":{\"h\":4,\"w\":6,\"x\":0,\"y\":0},\"title\":\"Total Predictions\",\"type\":\"stat\",\"datasource\":{\"type\":\"prometheus\",\"uid\":\"$DS_UID\"},\"targets\":[{\"expr\":\"sum(http_requests_total{endpoint=\\\"/predict\\\"})\",\"legendFormat\":\"Predictions\"}],\"options\":{\"reduceOptions\":{\"calcs\":[\"lastNotNull\"]},\"colorMode\":\"background\"},\"fieldConfig\":{\"defaults\":{\"color\":{\"mode\":\"fixed\",\"fixedColor\":\"blue\"}}}},
        {\"id\":2,\"gridPos\":{\"h\":4,\"w\":6,\"x\":6,\"y\":0},\"title\":\"Avg Inference Latency\",\"type\":\"stat\",\"datasource\":{\"type\":\"prometheus\",\"uid\":\"$DS_UID\"},\"targets\":[{\"expr\":\"rate(http_request_duration_seconds_sum{endpoint=\\\"/predict\\\"}[5m]) / rate(http_request_duration_seconds_count{endpoint=\\\"/predict\\\"}[5m])\",\"legendFormat\":\"Latency\"}],\"options\":{\"reduceOptions\":{\"calcs\":[\"lastNotNull\"]},\"colorMode\":\"background\"},\"fieldConfig\":{\"defaults\":{\"unit\":\"s\",\"color\":{\"mode\":\"thresholds\"},\"thresholds\":{\"steps\":[{\"color\":\"green\",\"value\":null},{\"color\":\"yellow\",\"value\":0.1},{\"color\":\"red\",\"value\":0.2}]}}}},
        {\"id\":3,\"gridPos\":{\"h\":4,\"w\":6,\"x\":12,\"y\":0},\"title\":\"Success Rate\",\"type\":\"stat\",\"datasource\":{\"type\":\"prometheus\",\"uid\":\"$DS_UID\"},\"targets\":[{\"expr\":\"sum(rate(http_requests_total{endpoint=\\\"/predict\\\",status=\\\"200\\\"}[5m])) / sum(rate(http_requests_total{endpoint=\\\"/predict\\\"}[5m]))\",\"legendFormat\":\"Success\"}],\"options\":{\"reduceOptions\":{\"calcs\":[\"lastNotNull\"]},\"colorMode\":\"background\"},\"fieldConfig\":{\"defaults\":{\"unit\":\"percentunit\",\"color\":{\"mode\":\"fixed\",\"fixedColor\":\"green\"}}}},
        {\"id\":4,\"gridPos\":{\"h\":4,\"w\":6,\"x\":18,\"y\":0},\"title\":\"Total API Requests\",\"type\":\"stat\",\"datasource\":{\"type\":\"prometheus\",\"uid\":\"$DS_UID\"},\"targets\":[{\"expr\":\"sum(http_requests_total)\",\"legendFormat\":\"Total\"}],\"options\":{\"reduceOptions\":{\"calcs\":[\"lastNotNull\"]},\"colorMode\":\"background\"},\"fieldConfig\":{\"defaults\":{\"color\":{\"mode\":\"fixed\",\"fixedColor\":\"purple\"}}}},
        {\"id\":5,\"gridPos\":{\"h\":8,\"w\":12,\"x\":0,\"y\":4},\"title\":\"Request Rate by Endpoint\",\"type\":\"timeseries\",\"datasource\":{\"type\":\"prometheus\",\"uid\":\"$DS_UID\"},\"targets\":[{\"expr\":\"rate(http_requests_total{endpoint=\\\"/predict\\\"}[1m])\",\"legendFormat\":\"/predict\"},{\"expr\":\"rate(http_requests_total{endpoint=\\\"/health\\\"}[1m])\",\"legendFormat\":\"/health\"},{\"expr\":\"rate(http_requests_total{endpoint=\\\"/metrics\\\"}[1m])\",\"legendFormat\":\"/metrics\"}],\"fieldConfig\":{\"defaults\":{\"unit\":\"reqps\"}}},
        {\"id\":6,\"gridPos\":{\"h\":8,\"w\":12,\"x\":12,\"y\":4},\"title\":\"Inference Latency Percentiles\",\"type\":\"timeseries\",\"datasource\":{\"type\":\"prometheus\",\"uid\":\"$DS_UID\"},\"targets\":[{\"expr\":\"histogram_quantile(0.50,rate(http_request_duration_seconds_bucket{endpoint=\\\"/predict\\\"}[5m]))\",\"legendFormat\":\"p50\"},{\"expr\":\"histogram_quantile(0.95,rate(http_request_duration_seconds_bucket{endpoint=\\\"/predict\\\"}[5m]))\",\"legendFormat\":\"p95\"},{\"expr\":\"histogram_quantile(0.99,rate(http_request_duration_seconds_bucket{endpoint=\\\"/predict\\\"}[5m]))\",\"legendFormat\":\"p99\"}],\"fieldConfig\":{\"defaults\":{\"unit\":\"s\"}}},
        {\"id\":7,\"gridPos\":{\"h\":6,\"w\":24,\"x\":0,\"y\":12},\"title\":\"HTTP Status Codes\",\"type\":\"timeseries\",\"datasource\":{\"type\":\"prometheus\",\"uid\":\"$DS_UID\"},\"targets\":[{\"expr\":\"sum by (status) (rate(http_requests_total[1m]))\",\"legendFormat\":\"{{status}}\"}],\"fieldConfig\":{\"defaults\":{\"unit\":\"reqps\"}}}
      ]
    }
  }" > /dev/null

echo "[grafana-setup] Dashboard ready at $GRAFANA/d/medical-imaging-v1"
