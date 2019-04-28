from django.conf.urls import url
from django.views.generic import TemplateView
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views

from customer_segmentation import views

app_name="portal"

urlpatterns = [
    url("^cust-seg-dashboard$", views.CustomerSegmentDashboard.as_view(), name='cust_seg_dashboard'),
    url("^data-preview$", views.DataPreview.as_view(), name='data_preview_page'),
    url("^cluster-detail$", views.cluster_details, name='cluster_details')
]