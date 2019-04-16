from django.shortcuts import render
import pandas as pd
from customer_segmentation.SegmentData import SegmentData

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
