# Scoring service image. Deliberately excludes Spark: the pipeline runs on
# Databricks or a laptop, while this image only loads the persisted model
# and serves predictions. The model artefact is mounted at runtime rather
# than baked in, so retraining never means rebuilding.

FROM python:3.13-slim AS build
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist


FROM python:3.13-slim
LABEL org.opencontainers.image.title="olist-retention-scoring" \
      org.opencontainers.image.source="https://github.com/ariyan-tipu/olist-customer-retention"

RUN useradd --create-home --uid 1000 scorer
WORKDIR /app

COPY --from=build /dist/*.whl /tmp/
RUN WHEEL=$(ls /tmp/*.whl) \
    && pip install --no-cache-dir "${WHEEL}[service]" \
    && rm -rf /tmp/*.whl

ENV OLIST_MODEL_PATH=/models/churn_model.joblib
EXPOSE 8000
USER scorer

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD \
    python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "--factory", "olist_retention.service.app:create_app", \
     "--host", "0.0.0.0", "--port", "8000"]
