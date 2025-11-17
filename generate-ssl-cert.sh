#!/bin/bash

# Script to generate self-signed SSL certificates for the face recognition service
# This creates certificates valid for 365 days

set -e

echo "=== Generating Self-Signed SSL Certificates ==="
echo ""

# Create nginx/ssl directory if it doesn't exist
mkdir -p nginx/ssl

# Get the server IP address (for Oracle VPS)
echo "Please enter your Oracle VPS IP address (or press Enter to use 0.0.0.0):"
read SERVER_IP
SERVER_IP=${SERVER_IP:-0.0.0.0}

echo ""
echo "Generating SSL certificate for IP: $SERVER_IP"
echo ""

# Generate private key
openssl genrsa -out nginx/ssl/key.pem 2048

# Generate certificate signing request (CSR) and certificate
openssl req -new -x509 -key nginx/ssl/key.pem -out nginx/ssl/cert.pem -days 365 \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=IT/CN=$SERVER_IP" \
    -addext "subjectAltName = IP:$SERVER_IP,IP:127.0.0.1,DNS:localhost"

# Set proper permissions
chmod 644 nginx/ssl/cert.pem
chmod 600 nginx/ssl/key.pem

echo ""
echo "=== SSL Certificates Generated Successfully ==="
echo ""
echo "Certificate location: nginx/ssl/cert.pem"
echo "Private key location: nginx/ssl/key.pem"
echo "Valid for: 365 days"
echo ""
echo "NOTE: This is a self-signed certificate. Your friend will need to:"
echo "1. Accept the security warning in their browser, OR"
echo "2. Add the certificate to their trusted certificates, OR"
echo "3. Use curl with -k flag: curl -k https://$SERVER_IP"
echo ""
echo "To view certificate details:"
echo "  openssl x509 -in nginx/ssl/cert.pem -text -noout"
echo ""
