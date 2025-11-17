# Quick Start - HTTPS Deployment

## One-Command Deployment

```bash
./deploy-https.sh
```

This script will:
1. Generate SSL certificates (if not present)
2. Build Docker images
3. Start all services
4. Verify the deployment

## Manual Deployment (Step by Step)

### 1. Generate SSL Certificates

```bash
./generate-ssl-cert.sh
```

Enter your Oracle VPS public IP when prompted.

### 2. Deploy Services

```bash
docker-compose up -d --build
```

### 3. Verify Deployment

```bash
./verify-setup.sh
```

### 4. Check Service Status

```bash
docker-compose ps
```

You should see:
- `shelia-nginx-proxy` - Running
- `shelia-face-recognition` - Running

## Testing Endpoints

### From Your VPS

```bash
# Health check
curl -k https://localhost/api/v1/health

# API Documentation
curl -k https://localhost/docs
```

### From External Machine (Your Friend's Laptop)

Replace `YOUR_VPS_IP` with your actual Oracle VPS public IP:

```bash
# Health check
curl -k https://YOUR_VPS_IP/api/v1/health

# Test face comparison
curl -k -X POST https://YOUR_VPS_IP/api/v1/compare-photos \
  -F "image1=@photo1.jpg" \
  -F "image2=@photo2.jpg"
```

### Browser Access

Visit: `https://YOUR_VPS_IP/docs`

**Note**: Accept the security warning (self-signed certificate)

## Oracle Cloud Firewall Setup

### Via Oracle Cloud Console

1. Go to Oracle Cloud Console
2. Navigate to: Networking → Virtual Cloud Networks
3. Select your VCN → Security Lists → Default Security List
4. Add Ingress Rules:
   - **Source CIDR**: 0.0.0.0/0
   - **IP Protocol**: TCP
   - **Destination Port Range**: 80

   - **Source CIDR**: 0.0.0.0/0
   - **IP Protocol**: TCP
   - **Destination Port Range**: 443

### Via VPS iptables

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## Common Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Nginx only
docker-compose logs -f nginx

# Face recognition only
docker-compose logs -f face-recognition
```

### Restart Services

```bash
docker-compose restart
```

### Stop Services

```bash
docker-compose down
```

### Rebuild After Code Changes

```bash
docker-compose down
docker-compose up -d --build
```

## Troubleshooting Quick Fixes

### Can't Connect from External Machine

1. Check Oracle Cloud security list (see above)
2. Check VPS firewall: `sudo iptables -L -n | grep -E '(80|443)'`
3. Verify public IP: `curl ifconfig.me`
4. Test from VPS first: `curl -k https://localhost/api/v1/health`

### 502 Bad Gateway

```bash
# Check face recognition service
docker-compose logs face-recognition

# Restart services
docker-compose restart
```

### SSL Certificate Errors

```bash
# Regenerate certificates
./generate-ssl-cert.sh
docker-compose restart nginx
```

## API Endpoints Reference

All endpoints use HTTPS:

- **Health**: `GET /api/v1/health`
- **Docs**: `GET /docs`
- **Compare Photos**: `POST /api/v1/compare-photos`
- **Extract Embedding**: `POST /api/v1/extract-embedding`

## Security Notes

- Self-signed certificate (browsers will show warning)
- No authentication (add if needed)
- CORS enabled for all origins
- Connection is encrypted despite certificate warning

## Need More Help?

See detailed guide: [HTTPS_DEPLOYMENT_GUIDE.md](HTTPS_DEPLOYMENT_GUIDE.md)
