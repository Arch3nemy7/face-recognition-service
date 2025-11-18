# Shelia Face Recognition Service - Docker Deployment Guide

This guide walks you through deploying the Shelia Face Recognition Service to your VPS using Docker and Docker Compose. The configuration is template-based, allowing you to customize it for your specific domain and VPS setup.

## Prerequisites

- VPS with Docker and Docker Compose installed
- Your own domain name (with DNS pointing to your VPS)
- SSH access to your VPS
- Basic familiarity with terminal/command line

## Quick Setup Steps

### 1. Clone/Copy the Repository

```bash
# Clone the repository to your VPS
git clone <repository-url> /path/to/shelia-face-recognition-service
cd /path/to/shelia-face-recognition-service
```

### 2. Create Environment Configuration

Copy the template environment file and customize it:

```bash
cp .env.example .env
```

Now edit `.env` and replace the template values with your own:

```bash
nano .env
# or use your preferred editor: vim, vi, code, etc.
```

### 3. Configure Your Domain

**Important:** You must update the Nginx configuration with your actual domain name.

Edit `nginx/nginx.conf`:

```bash
nano nginx/nginx.conf
```

Replace all instances of `{{DOMAIN_NAME}}` with your actual domain. For example:
- Before: `server_name {{DOMAIN_NAME}};`
- After: `server_name api.yourdomain.com;`

**Find and replace `{{DOMAIN_NAME}}` in these locations:**

1. **Line 53** - HTTP server block (certbot challenge)
2. **Line 71** - HTTPS server block
3. **Line 74** - SSL certificate path
4. **Line 75** - SSL certificate key path

### 4. Set Required Environment Variables

Edit `.env` and set these critical values:

#### Domain & SSL Configuration

```env
# Your actual domain name
DOMAIN_NAME=api.yourdomain.com

# Email for Let's Encrypt SSL certificate notifications
LETSENCRYPT_EMAIL=admin@yourdomain.com
```

#### Security - API Token

Generate a secure random token for API authentication:

```bash
# Generate a secure token
openssl rand -hex 32
```

Copy the output and set it in `.env`:

```env
API_TOKEN=<paste-your-generated-token-here>
```

#### Container Configuration

Customize based on your VPS:

```env
# Naming prefix for containers
CONTAINER_PREFIX=shelia

# Resource limits (adjust based on VPS capacity)
FACE_RECOGNITION_CPU_LIMIT=2        # CPU cores
FACE_RECOGNITION_MEMORY_LIMIT=2G    # Memory

FACE_RECOGNITION_CPU_RESERVATION=1  # Minimum reserved
FACE_RECOGNITION_MEMORY_RESERVATION=1G
```

#### Optional: CORS Settings

If deploying a frontend on the same domain:

```env
# Restrict CORS to specific origins for production
CORS_ORIGINS=["https://yourdomain.com", "https://www.yourdomain.com"]
```

### 5. Create Required Docker Network (if not exists)

```bash
# Create the external network for multi-service communication
docker network create proxy-network
```

### 6. Start the Services

```bash
# Start all services in the background
docker-compose up -d

# Check status
docker-compose ps
```

Monitor the certbot container to get SSL certificate:

```bash
# Watch certbot logs (wait until certificate is issued)
docker-compose logs -f certbot

# Expected output when successful:
# "Successfully received certificate"
```

This typically takes 2-5 minutes.

### 7. Verify Deployment

Once all containers are running:

```bash
# Check all services are healthy
docker-compose ps

# Test the API endpoint
curl -k https://your-domain.com/api/v1/health
```

You should receive a JSON health response.

## Configuration Options Explained

### Deployment Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `CONTAINER_PREFIX` | Prefix for all container names | `shelia` |
| `RESTART_POLICY` | Container restart behavior | `unless-stopped` |
| `INTERNAL_NETWORK_NAME` | Docker network for internal communication | `face-recognition-network` |
| `EXTERNAL_NETWORK_NAME` | External Docker network | `proxy-network` |

### Domain & SSL

| Variable | Description | Example |
|----------|-------------|---------|
| `DOMAIN_NAME` | Your actual domain | `api.yourdomain.com` |
| `LETSENCRYPT_EMAIL` | Email for SSL certificate | `admin@yourdomain.com` |

### API & Server

| Variable | Description | Default |
|----------|-------------|---------|
| `API_TOKEN` | Bearer token for API authentication | (required - generate new) |
| `DEBUG` | Enable debug mode | `false` |
| `LOG_LEVEL` | Logging level (debug, info, warning, error) | `info` |
| `DEVICE` | Compute device (cpu, cuda) | `cpu` |
| `MODEL_NAME` | Face recognition model (buffalo_l, buffalo_sc) | `buffalo_l` |

### Performance

| Variable | Description | Example |
|----------|-------------|---------|
| `FACE_RECOGNITION_CPU_LIMIT` | Max CPU cores | `2` |
| `FACE_RECOGNITION_MEMORY_LIMIT` | Max memory | `2G` |
| `FACE_RECOGNITION_CPU_RESERVATION` | Min CPU cores | `1` |
| `FACE_RECOGNITION_MEMORY_RESERVATION` | Min memory | `1G` |

## Troubleshooting

### Services Won't Start

Check logs:
```bash
docker-compose logs face-recognition
docker-compose logs nginx
docker-compose logs certbot
```

### SSL Certificate Issues

```bash
# Check certbot status
docker-compose logs certbot

# Manually renew certificate
docker-compose exec certbot certbot renew --dry-run
```

### Port Already in Use

If ports 80 or 443 are in use:

1. Check what's using them:
```bash
sudo netstat -tlnp | grep -E ':(80|443)'
# or
sudo lsof -i :80 -i :443
```

2. Stop the conflicting service or use different ports in `docker-compose.yml`

### API Not Responding

1. Verify container is running: `docker-compose ps`
2. Check health endpoint: `curl https://your-domain.com/api/v1/health`
3. Review logs: `docker-compose logs face-recognition`

## Common Tasks

### Update Environment Variables

Edit `.env`, then restart the service:

```bash
docker-compose up -d
```

### View Service Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f face-recognition
docker-compose logs -f nginx
```

### Stop Services

```bash
# Stop without removing
docker-compose stop

# Stop and remove
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v
```

### Change Domain

1. Update `DOMAIN_NAME` in `.env`
2. Update all `{{DOMAIN_NAME}}` instances in `nginx/nginx.conf`
3. Remove old SSL certificates: `rm -rf nginx/ssl/*`
4. Restart: `docker-compose up -d`

### Enable GPU Support (CUDA)

If your VPS has GPU:

1. Install NVIDIA Docker runtime on VPS
2. Update `.env`:
```env
DEVICE=cuda
```
3. Restart: `docker-compose up -d`

### Scale Resources

For higher load, edit `.env`:

```env
FACE_RECOGNITION_CPU_LIMIT=4
FACE_RECOGNITION_MEMORY_LIMIT=4G
FACE_RECOGNITION_CPU_RESERVATION=2
FACE_RECOGNITION_MEMORY_RESERVATION=2G
```

Then restart: `docker-compose up -d`

## File Structure

```
.
├── .env                          # Your configuration (create from .env.example)
├── .env.example                  # Configuration template
├── docker-compose.yml            # Docker Compose file with variables
├── Dockerfile                    # Python service Dockerfile
├── nginx/
│   ├── nginx.conf               # Nginx configuration (update with your domain)
│   ├── nginx.conf.template      # Reference template
│   ├── ssl/                     # SSL certificates (auto-generated by certbot)
│   └── certbot-webroot/         # Certbot challenge files
├── DEPLOYMENT_GUIDE.md          # This file
└── shelia_face_recognition_service/
    ├── main.py                  # Flask application
    ├── config.py                # Configuration loader
    └── auth.py                  # Authentication middleware
```

## Security Considerations

1. **API Token**: Generate a strong token using `openssl rand -hex 32`
2. **CORS**: Change `CORS_ORIGINS` from `["*"]` to specific domains in production
3. **SSL**: Ensure Let's Encrypt certificate is auto-renewing (verified in logs)
4. **Firewall**: Only expose ports 80 and 443 in your VPS firewall
5. **Rate Limiting**: Nginx is configured with rate limits for API endpoints

## Support & Next Steps

- Check the main README.md for API documentation
- Review `nginx/nginx.conf` for rate limiting and security headers
- Monitor logs regularly for issues: `docker-compose logs -f`
- Set up log rotation if running long-term

## Deployment Checklist

- [ ] Created `.env` from `.env.example`
- [ ] Updated `DOMAIN_NAME` in `.env` with your domain
- [ ] Updated `LETSENCRYPT_EMAIL` in `.env`
- [ ] Updated `DOMAIN_NAME` placeholders in `nginx/nginx.conf`
- [ ] Generated and set `API_TOKEN` in `.env`
- [ ] Created Docker network: `docker network create proxy-network`
- [ ] Started services: `docker-compose up -d`
- [ ] Verified SSL certificate is issued
- [ ] Tested health endpoint: `curl https://your-domain.com/api/v1/health`
- [ ] Reviewed logs for errors: `docker-compose logs`
- [ ] Set up monitoring/logging
