SERVER_HOSTNAME="ec2-3-147-104-220.us-east-2.compute.amazonaws.com"

# Remove trailing / if it exists
SERVER_HOSTNAME=${SERVER_HOSTNAME%/}

# make sure no dockers are alredy running
docker stop $(docker ps -aq)
# make sure flask is installed
sudo apt install python3-flask

docker start gitlab
docker start shopping
docker start shopping_admin
docker start forum
docker start kiwix33
cd /home/ubuntu/openstreetmap-website/
docker compose start

sleep 90

# available ports on toolkit:
#   - port: "22"
#   - port: "80"
#   - port: "443"
#   - port: "3306"
#   - port: "7565"
#   - port: "8080"
#   - port: "9001"
#   - port: "42022"

SHOPPING_OG_PORT=7770
SHOPPING_PORT=3306

SHOPPING_ADMIN_OG_PORT=7780
SHOPPING_ADMIN_PORT=7565

GITLAB_OG_PORT=8023
GITLAB_PORT=8080

FORUM_OG_PORT=9999
FORUM_PORT=9001

MAP_OG_PORT=3000
MAP_PORT=22

WIKIPEDIA_OG_PORT=8888
WIKIPEDIA_PORT=80

HOMEPAGE_PORT=42022

docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$SERVER_HOSTNAME:$SHOPPING_OG_PORT" # no trailing /
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value='http://$SERVER_HOSTNAME:$SHOPPING_OG_PORT/' WHERE path = 'web/secure/base_url';"
docker exec shopping /var/www/magento2/bin/magento cache:flush

docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0

docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$SERVER_HOSTNAME:$SHOPPING_ADMIN_OG_PORT" # no trailing "/
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value='http://$SERVER_HOSTNAME:$SHOPPING_ADMIN_OG_PORT/' WHERE path = 'web/secure/base_url';"
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

docker exec gitlab sed -i "s|^external_url.*|external_url 'http://$SERVER_HOSTNAME:$GITLAB_OG_PORT'|" /etc/gitlab/gitlab.rb
docker exec gitlab gitlab-ctl reconfigure

# redirect the forum to the correct port
sudo iptables -t nat -A PREROUTING -p tcp --dport $SHOPPING_PORT -j REDIRECT --to-port $SHOPPING_OG_PORT
sudo iptables -t nat -A PREROUTING -p tcp --dport $SHOPPING_ADMIN_PORT -j REDIRECT --to-port $SHOPPING_ADMIN_OG_PORT
sudo iptables -t nat -A PREROUTING -p tcp --dport $GITLAB_PORT -j REDIRECT --to-port $GITLAB_OG_PORT
sudo iptables -t nat -A PREROUTING -p tcp --dport $FORUM_PORT -j REDIRECT --to-port $FORUM_OG_PORT
sudo iptables -t nat -A PREROUTING -p tcp --dport $MAP_PORT -j REDIRECT --to-port $MAP_OG_PORT
sudo iptables -t nat -A PREROUTING -p tcp --dport $WIKIPEDIA_PORT -j REDIRECT --to-port $WIKIPEDIA_OG_PORT


# HOMEPAGE
cd /home/ubuntu/
perl -pi -e "s|<your-server-hostname>|${SERVER_HOSTNAME}|g" webarena/environment_docker/webarena-homepage/templates/index.html
cd webarena/environment_docker/webarena-homepage
flask run --host=0.0.0.0 --port=$HOMEPAGE_PORT
