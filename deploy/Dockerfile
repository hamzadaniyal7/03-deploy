# The gateway has no dependencies, so this is a small image on purpose:
# the thing in front of your inference servers should not be the part that
# needs patching every week.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Runs as a non-root user; nothing here needs more.
RUN useradd --create-home --uid 10001 selfserve
USER selfserve

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
      sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=3).status==200 else 1)"

ENTRYPOINT ["selfserve", "gateway"]
