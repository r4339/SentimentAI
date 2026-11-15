
#!/bin/bash

set -e

echo "Starting SentimentAI deployment..."

docker compose -f deploy/compose.yaml pull

docker compose -f deploy/compose.yaml up -d

echo "Deployment completed successfully!"
