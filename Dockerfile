# Uniform evaluation environment for the EuroSys'27 artifact.
#
# Build and run the smoke test:
#     docker build -t routewise-ae .
#     docker run --rm routewise-ae
# Interactive shell for the other commands in the README:
#     docker run --rm -it routewise-ae bash
#
# The base image and the uv binary are version-pinned; uv installs the
# Python interpreter pinned by .python-version and the exact locked
# dependency set from uv.lock.

FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

WORKDIR /artifact
COPY . .
RUN uv sync --frozen

CMD ["bash", "scripts/artifact_smoke_test.sh"]
