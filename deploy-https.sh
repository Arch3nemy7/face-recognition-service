#!/bin/bash

# Quick deployment script for HTTPS Face Recognition Service
# This script automates the entire deployment process

set -e

echo "=========================================="
echo "  Face Recognition Service HTTPS Deployment"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Step 1: Check for existing SSL certificates
echo "Step 1: Checking SSL certificates..."
if [ -f "nginx/ssl/cert.pem" ] && [ -f "nginx/ssl/key.pem" ]; then
    echo "SSL certificates found."
    read -p "Do you want to regenerate them? (y/N): " REGEN
    if [ "$REGEN" = "y" ] || [ "$REGEN" = "Y" ]; then
        ./generate-ssl-cert.sh
    fi
else
    echo "SSL certificates not found. Generating..."
    ./generate-ssl-cert.sh
fi

echo ""
echo "Step 2: Stopping any existing containers..."
docker-compose down 2>/dev/null || true

echo ""
echo "Step 3: Building and starting services..."
docker-compose up -d --build

echo ""
echo "Step 4: Waiting for services to start..."
sleep 10

# Check if containers are running
echo ""
echo "Step 5: Verifying deployment..."
if docker ps | grep -q "shelia-nginx-proxy" && docker ps | grep -q "shelia-face-recognition"; then
    echo "✓ All containers are running successfully!"
else
    echo "✗ Some containers failed to start. Checking logs..."
    docker-compose logs --tail=50
    exit 1
fi

# Test health endpoint
echo ""
echo "Step 6: Testing health endpoint..."
sleep 5
HEALTH_CHECK=$(curl -k -s https://localhost/api/v1/health 2>/dev/null || echo "failed")
if echo "$HEALTH_CHECK" | grep -q "status"; then
    echo "✓ Health check passed!"
    echo "Response: $HEALTH_CHECK"
else
    echo "⚠ Health check did not return expected response"
    echo "Response: $HEALTH_CHECK"
fi

# Get public IP
echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""

# Try to get public IP
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_VPS_IP")

echo "Your face recognition service is now running with HTTPS!"
echo ""
echo "Access URLs:"
echo "  • API Docs: https://$PUBLIC_IP/docs"
echo "  • Health Check: https://$PUBLIC_IP/api/v1/health"
echo ""
echo "Container Status:"
docker-compose ps
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop the service:"
echo "  docker-compose down"
echo ""
echo "For your friend to test from their laptop:"
echo "  curl -k https://$PUBLIC_IP/api/v1/health"
echo ""
echo "NOTE: Users will see a security warning due to self-signed certificate."
echo "This is normal. The connection is still encrypted."
echo ""
