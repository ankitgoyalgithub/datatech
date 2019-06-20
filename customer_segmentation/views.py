import csv
import json
import logging
import pandas as pd

from django.conf import settings
from django.shortcuts import render
from django.views import generic
from django.http import HttpResponseRedirect,HttpResponse, HttpResponseNotFound

from rest_framework.decorators import api_view
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser

from customer_segmentation.SegmentData import SegmentData

logger = logging.getLogger(__name__)

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

class DataPreview(generic.ListView):
    template_name = "data_preview.html"
    context_object_name = "data"

    def get_queryset(self):
        data = dict()
        media_url = settings.MEDIA_ROOT
        preview_file = open(media_url + '/data_files/pageview.csv')
        field_names = ("ACCOUNT_ID", "AVERAGE_TIME_ON_PAGE", "EXITS", "PAGEVIEWS")
        reader = csv.DictReader(preview_file, field_names)
        data_rows = [row for row in reader]
        del data_rows[0]
        data['cols'] = list(field_names)
        data['rows'] = data_rows
        return data

class AlgorithmsSelection(generic.ListView):
    template_name = "select_algorithm.html"
    context_object_name = "data"

    def get_queryset(self):
        data = dict()
        data['algos'] = ['a1','a2','a3']
        return data

"""
API to Fetch the Data Set on Which Clustering Will be Performed
"""
@api_view(['GET'])
def data_preview(request):
    try:
        media_url = settings.MEDIA_ROOT
        preview_file = open(media_url + '/data_files/pageview.csv')
        field_names = ("ACCOUNT_ID", "AVERAGE_TIME_ON_PAGE", "EXITS", "PAGEVIEWS")
        reader = csv.DictReader(preview_file, field_names)
        out = json.dumps([row for row in reader])
        return JSONResponse(out)
    except Exception as e:
        logger.error(e)
        return JSONResponse(json.dumps({"error": "Unable to Generate Preview"}))

"""
API to Get the Distribution of Each Cluster.
"""
@api_view(['GET'])
def cluster_distribution(request):
    try:
        media_url = settings.MEDIA_ROOT
        input_df = pd.read_csv(media_url + '/data_files/pageview.csv')
        data = dict()
        obj=SegmentData(input_df,['PAGEVIEWS','EXITS'],['AVERAGE_TIME_ON_PAGE'],'ACCOUNT_ID',3,'N')
        rank = obj.run_segment()[0]
        for index, row in rank.iterrows():
            data["C" + str(index+1)] =  row['Cluster_Size%']
        return JSONResponse(data)
    except Exception as e:
        logger.error(e)
        return JSONResponse(json.dumps({"error": "Unable to Fetch Cluster Distribution Data"}))

"""
API to Get the Details of Each Cluster.
"""
@api_view(['GET'])
def cluster_details(request):
    try:
        media_url = settings.MEDIA_ROOT
        input_df = pd.read_csv(media_url + '/data_files/pageview.csv')
        data = dict()
        obj=SegmentData(input_df,['PAGEVIEWS','EXITS'],['AVERAGE_TIME_ON_PAGE'],'ACCOUNT_ID',3,'N')
        rank = obj.run_segment()[0]
        data["rows"] = list()
        data["columns"] = list()
        
        for index, row in rank.iterrows():
            temp_details = dict()
            temp_details['LABEL'] = index
            temp_details['COUNT'] =  row['Cluster_Size_Count']
            temp_details['AVERAGE_TIME_ON_PAGE'] =  row['AVERAGE_TIME_ON_PAGE']
            temp_details['PAGEVIEWS'] =  row['PAGEVIEWS']
            temp_details['EXITS'] =  row['EXITS']
            temp_details['AVERAGE_TIME_ON_PAGE_LEVEL'] =  row['AVERAGE_TIME_ON_PAGE_Level']
            temp_details['EXITS_LEVEL'] =  row['EXITS_Level']
            temp_details['PAGEVIEWS_LEVEL'] =  row['PAGEVIEWS_Level']
            data["rows"].append(temp_details)

        data["columns"] = [
            "LABEL", 
            "COUNT", 
            "AVERAGE_TIME_ON_PAGE",
            "PAGEVIEWS",
            "EXITS",
            "AVERAGE_TIME_ON_PAGE_LEVEL",
            "EXITS_LEVEL",
            "PAGEVIEWS_LEVEL"
            ]
        return JSONResponse(data)
    except Exception as e:
        logger.error(e)
        raise e

"""
API to Get the Details of Each Cluster.
"""
@api_view(['GET'])
def account_insights(request):
    try:
        media_url = settings.MEDIA_ROOT
        input_df = pd.read_csv(media_url + '/data_files/pageview.csv')
        data = dict()
        obj=SegmentData(input_df,['PAGEVIEWS','EXITS'],['AVERAGE_TIME_ON_PAGE'],'ACCOUNT_ID',3,'N')
        account_insights = obj.run_segment()[1]

        data["rows"] = list()
        data["columns"] = list()
        for index, row in account_insights.iterrows():
            temp_details = dict()
            temp_details['LABEL'] = index
            temp_details['ACCOUNTID'] =  row['ACCOUNT_ID']
            temp_details['AVERAGE_TIME_ON_PAGE'] =  row['AVERAGE_TIME_ON_PAGE']
            temp_details['PAGEVIEWS'] =  row['PAGEVIEWS']
            temp_details['CLUSTER'] =  row['clusterLabelColumn']
            temp_details['PAGEVIEWS_LEVEL'] =  row['PAGEVIEWS_Level']
            temp_details['EXITS_LEVEL'] =  row['EXITS_Level']
            temp_details['MESSAGE'] =  row['message']
            data["rows"].append(temp_details)
        
        data["columns"] = ["LABEL", "ACCOUNTID", "AVERAGE_TIME_ON_PAGE", "PAGEVIEWS", "CLUSTER", "PAGEVIEWS_LEVEL", "EXITS_LEVEL", "MESSAGE"]
        return JSONResponse(data)
    except Exception as e:
        logger.error(e)
        raise e

"""
API to Get the Details of Each Cluster.
"""
@api_view(['GET'])
def dist_plot(request):
    try:
        media_url = settings.MEDIA_ROOT
        input_df = pd.read_csv(media_url + '/data_files/pageview.csv')
        data = dict()
        obj=SegmentData(input_df,['PAGEVIEWS','EXITS'],['AVERAGE_TIME_ON_PAGE'],'ACCOUNT_ID',3,'N')
        account_insights = obj.run_segment()[1]

        data["rows"] = list()
        data["columns"] = list()
        for index, row in account_insights.iterrows():
            temp_details = dict()
            temp_details['LABEL'] = index
            temp_details['ACCOUNTID'] =  row['ACCOUNT_ID']
            temp_details['AVERAGE_TIME_ON_PAGE'] =  row['AVERAGE_TIME_ON_PAGE']
            temp_details['PAGEVIEWS'] =  row['PAGEVIEWS']
            temp_details['CLUSTER'] =  row['clusterLabelColumn']
            temp_details['PAGEVIEWS_LEVEL'] =  row['PAGEVIEWS_Level']
            temp_details['EXITS_LEVEL'] =  row['EXITS_Level']
            temp_details['MESSAGE'] =  row['message']
            data["rows"].append(temp_details)
        
        data["columns"] = ["LABEL", "ACCOUNTID", "AVERAGE_TIME_ON_PAGE", "PAGEVIEWS", "CLUSTER", "PAGEVIEWS_LEVEL", "EXITS_LEVEL", "MESSAGE"]
        return JSONResponse(data)
    except Exception as e:
        logger.error(e)
        raise e

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
