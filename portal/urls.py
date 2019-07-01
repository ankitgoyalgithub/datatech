from django.conf.urls import url
from django.views.generic import TemplateView
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views

from portal import views

app_name="portal"

urlpatterns = [
    url("^data-discovery$", TemplateView.as_view(template_name='data-discovery.html'), name='data_discovery'),
    url("^portfolio$", TemplateView.as_view(template_name='portfolio.html'), name='portfolio'),
    url("^blog$", TemplateView.as_view(template_name='blog.html'), name='blog'),
    url("^tour$", TemplateView.as_view(template_name='tour.html'), name='tour'),
    url("^contact$", TemplateView.as_view(template_name='contact.html'), name='contact'),
    url("^pricing$", TemplateView.as_view(template_name='pricing.html'), name='pricing'),
    url("^churn$", TemplateView.as_view(template_name='churn.html'), name='churn'),
    url("", TemplateView.as_view(template_name='index.html'), name='index')
]