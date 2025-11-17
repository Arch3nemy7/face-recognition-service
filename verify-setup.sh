#!/bin/bash

# Verification script for HTTPS Face Recognition Service
# This script tests the endpoints and validates the setup

set -e

echo "=== Face Recognition Service HTTPS Verification ==="
echo ""

# Get server address
echo "Enter your server IP address (or press Enter for localhost):"
read SERVER_IP
SERVER_IP=${SERVER_IP:-localhost}

echo ""
echo "Testing connection to: https://$SERVER_IP"
echo ""

# Test 1: Check if SSL certificates exist
echo "1. Checking SSL certificates..."
if [ -f "nginx/ssl/cert.pem" ] && [ -f "nginx/ssl/key.pem" ]; then
    echo "   ✓ SSL certificates found"
else
    echo "   ✗ SSL certificates not found. Run ./generate-ssl-cert.sh first"
    exit 1
fi

# Test 2: Check if containers are running
echo ""
echo "2. Checking Docker containers..."
if docker ps | grep -q "shelia-nginx-proxy"; then
    echo "   ✓ Nginx reverse proxy is running"
else
    echo "   ✗ Nginx reverse proxy is not running"
fi

if docker ps | grep -q "shelia-face-recognition"; then
    echo "   ✓ Face recognition service is running"
else
    echo "   ✗ Face recognition service is not running"
fi

# Test 3: Test HTTP redirect to HTTPS
echo ""
echo "3. Testing HTTP to HTTPS redirect..."
HTTP_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -L http://$SERVER_IP/api/v1/health 2>/dev/null || echo "000")
if [ "$HTTP_RESPONSE" = "200" ]; then
    echo "   ✓ HTTP redirects to HTTPS successfully"
else
    echo "   ⚠ HTTP redirect test returned code: $HTTP_RESPONSE"
fi

# Test 4: Test HTTPS health endpoint
echo ""
echo "4. Testing HTTPS health endpoint..."
HEALTH_RESPONSE=$(curl -k -s https://$SERVER_IP/api/v1/health 2>/dev/null || echo "failed")
if echo "$HEALTH_RESPONSE" | grep -q "status"; then
    echo "   ✓ HTTPS health endpoint is working"
    echo "   Response: $HEALTH_RESPONSE"
else
    echo "   ✗ HTTPS health endpoint failed"
    echo "   Response: $HEALTH_RESPONSE"
fi

# Test 5: Test HTTPS docs endpoint
echo ""
echo "5. Testing HTTPS API documentation..."
DOCS_RESPONSE=$(curl -k -s -o /dev/null -w "%{http_code}" https://$SERVER_IP/docs 2>/dev/null || echo "000")
if [ "$DOCS_RESPONSE" = "200" ]; then
    echo "   ✓ API documentation accessible at https://$SERVER_IP/docs"
else
    echo "   ⚠ API documentation returned code: $DOCS_RESPONSE"
fi

# Display certificate info
echo ""
echo "6. SSL Certificate Information:"
echo ""
openssl x509 -in nginx/ssl/cert.pem -noout -subject -issuer -dates 2>/dev/null || echo "   Could not read certificate"

# Final summary
echo ""
echo "=== Verification Complete ==="
echo ""
echo "Access your service at:"
echo "  • HTTPS: https://$SERVER_IP"
echo "  • API Docs: https://$SERVER_IP/docs"
echo "  • Health Check: https://$SERVER_IP/api/v1/health"
echo ""
echo "For your friend to access from another machine:"
echo "  1. Use your public IP address instead of localhost"
echo "  2. Ensure firewall allows ports 80 and 443"
echo "  3. They may need to accept the self-signed certificate warning"
echo ""
echo "Test command for your friend:"
echo "  curl -k https://$SERVER_IP/api/v1/health"
echo ""
