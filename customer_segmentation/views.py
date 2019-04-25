import csv
import json
import pandas as pd

from django.conf import settings
from django.shortcuts import render
from django.views import generic
from django.http import HttpResponseRedirect,HttpResponse, HttpResponseNotFound

from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser

from customer_segmentation.SegmentData import SegmentData

class JSONResponse(HttpResponse):
    def __init__(self, data, **kwargs):
        content = JSONRenderer().render(data)
        kwargs['content_type'] = 'application/json'
        super(JSONResponse, self).__init__(content, **kwargs)

class CustomerSegmentDashboard(generic.ListView):
    template_name = "customer_segmentation.html"
    context_object_name = "data"

    def get_queryset(self):
        data = dict()
        return data

"""
API to Fetch the Data Set on Which Clustering Will be Performed
"""
def data_preview(request):
    media_url = settings.MEDIA_ROOT
    preview_file = open(media_url + '/data_files/pageview.csv')
    field_names = ("ACCOUNT_ID", "AVERAGE_TIME_ON_PAGE", "EXITS", "PAGEVIEWS")
    reader = csv.DictReader(preview_file, field_names)
    out = json.dumps([row for row in reader])
    return JSONResponse(out)


"""
API to Get the Distribution of Each Cluster.
"""
def cluster_details(request):
    data =  {
        "c1": 35,
        "C2": 17,
        "C3": 48
    }
    return JSONResponse(data)

# Create your views here.
def input_method(input_dict):
    try:
        dataframe = pd.read_csv(input_dict['data'])
        seg_col = input_dict['seg_col']
        ob_col = input_dict['ob_col']
        ID = input_dict['ID']
        n_cluster = input_dict['n_cluster']
        optimizer = input_dict['optimizer']
        segment_obj = SegmentData(dataframe, seg_col, ob_col, ID, n_cluster, optimizer)
        rank, account_insights, cluster_insights, cluster_size, accuracy = segment_obj.run_segment()
        result_dict={
            'rank':rank,
            'account_insights':account_insights,
            'cluster_insights':cluster_insights,
            'cluster_size':cluster_size,
            'accuracy':accuracy
        }
        return result_dict
    except Exception as e:
        raise e

if __name__=='__main__':
    df = '/Users/mkumar/Desktop/datatech/pageview.csv'
    print("=======")
    input_map={'data':df,
               'seg_col':['Pageviews', 'Exits'],
               'ob_col': ['Avg. Time on Page'],
               'ID':'Account Id',
               'n_cluster': 3,
               'optimizer':'N'
               }

    # data_frame, seg_col, ID=[], n_cluster=5,optimize_no_cluster='Y'):
    #obj = SegmentData(df, ['Pageviews', 'Exits'], ['Avg. Time on Page'], 'Account Id', 3, 'N')
    #rank, account_insights, cluster_insights, cluster_size, accuracy = obj.run_segment()
    result = input_method(input_map)
    print (result)
