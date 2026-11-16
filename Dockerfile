FROM python:3.12-slim-bookworm
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY fedrag/ ./fedrag/
COPY chaincode/ ./chaincode/
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 7050 8443
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
