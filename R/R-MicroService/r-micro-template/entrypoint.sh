#!/bin/bash
set -e
set -x

## RUN R APP ==================
R_CMD="Rscript /app/api.R"
echo "Running R Script"
${R_CMD}
