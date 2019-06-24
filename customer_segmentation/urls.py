from django.conf.urls import url
from django.views.generic import TemplateView
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views

from customer_segmentation import views

app_name="portal"

urlpatterns = [
    url("^cust-seg-dashboard$", views.CustomerSegmentDashboard.as_view(), name='cust_seg_dashboard'),
    url("^file-upload$", views.FileUpload.as_view(), name='file_upload'),
    url("^data-preview$", views.DataPreview.as_view(), name='data_preview_page'),
    url("^select-algorithm", views.AlgorithmsSelection.as_view(), name="select_algorithms"),
    url("^cluster-distribution$", views.cluster_distribution, name='cluster_distribution'),
    url("^cluster-details$", views.cluster_details, name='cluster_details'),
    url("^account-insights$", views.account_insights, name='account_insights'),
    url("^upload-file$", views.upload_file, name='upload_file')
]