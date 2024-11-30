#!/bin/bash

# Configuration
DOMAIN="your-domain.com"  # Replace with your actual domain
EMAIL="your-email@example.com"  # Replace with your email

# Install required packages
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx

# Install SSL certificate
sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos -m $EMAIL

# Copy nginx configuration
sudo cp nginx.conf /etc/nginx/sites-available/$DOMAIN
sudo ln -s /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Create web root directory
sudo mkdir -p /var/www/sector-h
sudo chown -R $USER:$USER /var/www/sector-h

# Copy application files
sudo cp -r ../api /var/www/sector-h/
sudo cp -r ../frontend/dist/* /var/www/sector-h/

# Set up Python environment
cd /var/www/sector-h
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start application server
sudo systemctl start nginx
sudo systemctl enable nginx

# Setup firewall
sudo ufw allow 'Nginx Full'
sudo ufw allow ssh

echo "Deployment completed! Your site should be accessible at https://$DOMAIN"
