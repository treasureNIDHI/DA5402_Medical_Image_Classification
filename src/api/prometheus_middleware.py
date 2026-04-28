from prometheus_client import Counter, Histogram
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_errors_total = Counter(
    "http_errors_total",
    "Total failed requests",
    ["method", "endpoint", "status"]
)

request_latency = Histogram(
    "request_latency_seconds",
    "Request latency",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        method = request.method
        endpoint = request.url.path

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            latency = time.time() - start_time

            http_requests_total.labels(method, endpoint, str(status)).inc()
            http_errors_total.labels(method, endpoint, str(status)).inc()
            request_latency.labels(method, endpoint).observe(latency)

            raise

        latency = time.time() - start_time

        http_requests_total.labels(method, endpoint, str(status)).inc()

        if status >= 400:
            http_errors_total.labels(method, endpoint, str(status)).inc()

        request_latency.labels(method, endpoint).observe(latency)

        return response


def add_prometheus_middleware(app):
    app.add_middleware(PrometheusMiddleware)