from django.conf.urls import url
from django.views.generic import TemplateView
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views

from portal import views

app_name="portal"

urlpatterns = [
    url("^portfolio$", TemplateView.as_view(template_name='portfolio.html'), name='portfolio'),
    url("^tour$", TemplateView.as_view(template_name='tour.html'), name='tour'),
    url("^contact$", TemplateView.as_view(template_name='contact.html'), name='contact'),
    url("", TemplateView.as_view(template_name='index.html'), name='index')
]