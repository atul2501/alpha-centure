#!/usr/bin/env bash
# Nightly pg_dump to S3. Needs S3_BACKUP_URI in /etc/alpha/.env and an EC2 instance role with s3:PutObject.
set -euo pipefail
set -a; source /etc/alpha/.env; set +a
if [ -z "${S3_BACKUP_URI:-}" ]; then echo "S3_BACKUP_URI not set, skipping"; exit 0; fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
pg_dump -Fc "$DATABASE_URL" | aws s3 cp - "${S3_BACKUP_URI%/}/alpha_${STAMP}.dump"
echo "backup alpha_${STAMP}.dump uploaded"
