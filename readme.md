Steps To Deploy Changes:
1. source datatech/dtpr/bin/activate
2. git pull origin master
3. python manage.py collectstatic
4. python manage.py makemigrations
5. python manage.py migrate
6. sudo pkill -f uwsgi -9
7. uwsgi django.ini
8. sudo systemctl restart nginx