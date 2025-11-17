# HTTPS Deployment Guide for Face Recognition Service

This guide will help you deploy the Face Recognition Service with HTTPS using a reverse proxy on your Oracle VPS.

## Architecture Overview

```
Internet
    ↓
[Oracle VPS Firewall: Ports 80, 443]
    ↓
[Nginx Reverse Proxy]
    ├─ Port 80 (HTTP) → Redirects to HTTPS
    └─ Port 443 (HTTPS) → Proxies to Face Recognition Service
         ↓
[Face Recognition Service]
    └─ Port 8000 (Internal only)
```

## Prerequisites

- Oracle VPS with Docker and Docker Compose installed
- Root or sudo access
- Ports 80 and 443 open in your firewall

## Step 1: Configure Oracle VPS Firewall

On your Oracle VPS, ensure ports 80 and 443 are open:

```bash
# Using iptables
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

# OR using firewalld
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

Also ensure your Oracle Cloud security list allows ingress traffic on ports 80 and 443:
1. Go to Oracle Cloud Console
2. Navigate to your instance's subnet
3. Edit the Security List
4. Add Ingress Rules for ports 80 and 443 (TCP)

## Step 2: Generate SSL Certificates

Since you don't have a domain, we'll use self-signed certificates:

```bash
./generate-ssl-cert.sh
```

When prompted, enter your Oracle VPS public IP address. This will:
- Generate a private key (nginx/ssl/key.pem)
- Generate a certificate (nginx/ssl/cert.pem)
- Set appropriate permissions

## Step 3: Deploy the Service

Stop any existing containers:

```bash
docker-compose down
```

Build and start the services:

```bash
docker-compose up -d --build
```

This will start:
- Nginx reverse proxy (ports 80, 443)
- Face Recognition service (internal port 8000)

## Step 4: Verify the Deployment

Run the verification script:

```bash
./verify-setup.sh
```

Check that all services are running:

```bash
docker-compose ps
```

You should see both `shelia-nginx-proxy` and `shelia-face-recognition` running.

## Step 5: Test from Your VPS

Test the health endpoint:

```bash
# Using localhost
curl -k https://localhost/api/v1/health

# Using your VPS IP
curl -k https://YOUR_VPS_IP/api/v1/health
```

Expected response:
```json
{"status":"healthy","timestamp":"..."}
```

Access the API documentation:
```bash
# Open in browser (accept the security warning)
https://YOUR_VPS_IP/docs
```

## Step 6: Test from Your Friend's Laptop

Your friend needs to:

1. **Get your VPS public IP address** (let's say it's `123.45.67.89`)

2. **Test the health endpoint**:
   ```bash
   curl -k https://123.45.67.89/api/v1/health
   ```

3. **Access via browser**:
   - Visit `https://123.45.67.89/docs`
   - Accept the security warning (since it's a self-signed certificate)
   - They should see the API documentation

4. **Test the face recognition endpoint**:
   ```bash
   # Example: Compare two photos
   curl -k -X POST https://123.45.67.89/api/v1/compare-photos \
     -F "image1=@photo1.jpg" \
     -F "image2=@photo2.jpg"
   ```

## Understanding the Self-Signed Certificate Warning

Since we're using a self-signed certificate (no domain), users will see a security warning:
- **Chrome/Edge**: "Your connection is not private" - Click "Advanced" → "Proceed to [IP] (unsafe)"
- **Firefox**: "Warning: Potential Security Risk" - Click "Advanced" → "Accept the Risk and Continue"
- **curl**: Use the `-k` or `--insecure` flag

This is normal for self-signed certificates. The connection is still encrypted.

## Monitoring and Logs

View logs for all services:
```bash
docker-compose logs -f
```

View nginx logs only:
```bash
docker-compose logs -f nginx
```

View face recognition service logs:
```bash
docker-compose logs -f face-recognition
```

## Troubleshooting

### Problem: Connection refused

**Check 1**: Verify containers are running
```bash
docker-compose ps
```

**Check 2**: Verify ports are open
```bash
sudo netstat -tlnp | grep -E ':(80|443)'
```

**Check 3**: Check Oracle Cloud security list (see Step 1)

### Problem: SSL certificate errors

**Solution**: Regenerate certificates
```bash
./generate-ssl-cert.sh
docker-compose restart nginx
```

### Problem: 502 Bad Gateway

This means nginx can't reach the face recognition service.

**Check**: Ensure face-recognition service is running
```bash
docker-compose logs face-recognition
```

**Fix**: Restart services
```bash
docker-compose restart
```

### Problem: Can't access from external machine

**Check 1**: Verify you're using the public IP, not localhost

**Check 2**: Test from VPS first
```bash
curl -k https://localhost/api/v1/health
```

**Check 3**: Verify Oracle Cloud security list allows ports 80 and 443

**Check 4**: Verify VPS firewall rules
```bash
sudo iptables -L -n | grep -E '(80|443)'
```

## API Endpoints

All endpoints are now accessible via HTTPS:

- **Health Check**: `GET https://YOUR_IP/api/v1/health`
- **API Documentation**: `https://YOUR_IP/docs`
- **Compare Photos**: `POST https://YOUR_IP/api/v1/compare-photos`
- **Extract Embedding**: `POST https://YOUR_IP/api/v1/extract-embedding`

## Security Notes

1. **Self-signed certificate**: The connection is encrypted, but browsers will show warnings
2. **No authentication**: Consider adding API key authentication for production
3. **CORS enabled**: Service accepts requests from any origin (configured in docker-compose.yml)
4. **Rate limiting**: Consider adding rate limiting in nginx for production

## Updating the Service

To update the service:

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

## Stopping the Service

```bash
docker-compose down
```

To also remove volumes:
```bash
docker-compose down -v
```

## Production Recommendations

For a production setup, consider:

1. **Get a domain name** and use Let's Encrypt for free SSL certificates
2. **Add authentication** to protect your API
3. **Implement rate limiting** to prevent abuse
4. **Set up monitoring** (e.g., Prometheus + Grafana)
5. **Configure log rotation** to prevent disk space issues
6. **Use a CDN** if serving globally
7. **Set up automated backups**

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Review the logs: `docker-compose logs`
3. Verify all prerequisites are met
4. Ensure your Oracle Cloud security list is properly configured
