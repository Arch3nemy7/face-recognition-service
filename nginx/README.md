# Nginx Reverse Proxy Configuration

This directory contains the nginx configuration for the HTTPS reverse proxy.

## Directory Structure

```
nginx/
├── nginx.conf         # Main nginx configuration
├── ssl/              # SSL certificates directory
│   ├── cert.pem      # SSL certificate (generated)
│   └── key.pem       # Private key (generated)
└── README.md         # This file
```

## Configuration Overview

The `nginx.conf` file configures:

1. **HTTP Server (Port 80)**
   - Redirects all HTTP traffic to HTTPS

2. **HTTPS Server (Port 443)**
   - SSL/TLS termination
   - Reverse proxy to face-recognition service
   - Security headers
   - Gzip compression

3. **Upstream**
   - Routes requests to `face-recognition:8000` (Docker internal network)

## SSL Certificates

SSL certificates are stored in the `ssl/` subdirectory and are generated using the `generate-ssl-cert.sh` script in the project root.

**Important**: SSL certificates are gitignored for security reasons.

## Customization

### Increase Upload Size Limit

To allow larger image uploads, modify in `nginx.conf`:

```nginx
client_max_body_size 20M;  # Change this value
```

### Add Rate Limiting

Add this to the `http` block:

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

# Then in the server block:
limit_req zone=api_limit burst=20 nodelay;
```

### Custom Security Headers

Modify the security headers in the `server` block as needed for your use case.

## Testing

Test the nginx configuration:

```bash
docker-compose exec nginx nginx -t
```

Reload nginx after configuration changes:

```bash
docker-compose exec nginx nginx -s reload
```

## Logs

View nginx logs:

```bash
# Access logs
docker-compose logs nginx | grep "GET\|POST"

# Error logs
docker-compose logs nginx | grep "error"
```
