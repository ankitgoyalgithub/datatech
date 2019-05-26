import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sklearn.metrics as met
import jenkspy
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from pandas import Series

#from cluster_optimizer import optimize
#from celery.utils.log import get_task_logger

#LOGGER = get_task_logger(__name__)

# TODO
# Always aggregate on Account and optionally Account+Date
# Refactor account level Insights and any other for loops
# Outlier detection improvements
# Support for suppressing clusters

# Support for Segmentation Types
# Categorical Variables for observational variables
# Descriptive statistics of categorical variables
# Out of box segments for Health Score 2.0 both create and analyze, revenue

class SegmentData(object):
    def __init__(self, data_frame, seg_col,ob_col, ID=[], n_cluster=5,optimize_no_cluster='Y'):
        self.data_frame=data_frame
        self.seg_col=seg_col
        self.ob_col=ob_col
        self.ID=ID
        self.n_cluster = n_cluster
        self.optimize_no_cluster=optimize_no_cluster
        self.cluster_label_column= 'clusterLabelColumn'

    @staticmethod
    def transform_to_field_names(columns):
        return map(lambda column: SegmentationProject.get_field_id(column), columns)

    def _get_account_insight(self, data, rank, cluster_label):
        #LOGGER.info("rid :[%s] getting insights", self.request_id)
        measure_columns = self.seg_col or [] + self.ob_col or []
        level_columns = [measure_column + '_Level' for measure_column in measure_columns]

        rank_dict = rank.to_dict()

        def assign_levels(row):
            cluster_no = row[cluster_label]
            levels = [rank_dict[col][cluster_no] for col in level_columns]
            # levels = [rank.ix[cluster_no][col] for col in level_columns]
            message = ",".join(
                ["{} is {}".format(column.replace("__gc", "").replace("_", " "),
                                   levels[measure_columns.index(column)]) for column in measure_columns])
            levels.append(message)
            return tuple(levels)

        level_series = data[measure_columns + [self.cluster_label_column]].apply(assign_levels,
                                                                                 axis=1)
        level_df = pd.DataFrame(level_series.tolist(), columns=level_columns + ["message"])
        data.reset_index(inplace=True)
        merged_insight = pd.concat([data, level_df], axis=1)
        data.set_index(self.ID, inplace=True)
        return merged_insight

    def _insight(self, result):
        # no_of_groups = 3 if self.n_cluster < 5 else 5
        num_of_clusters = self.n_cluster
        if num_of_clusters == -1:
            num_of_clusters = result[self.cluster_label_column].nunique()
        if num_of_clusters < 3:
            no_of_groups = 2
        elif 3 <= num_of_clusters < 5:
            no_of_groups = 3
        else:
            no_of_groups = 5

        matrix = result.groupby(self.cluster_label_column).mean()
        for col in matrix.columns:
            rank = []
            
            level_col = []
            matrix_col = matrix[col].values if type(matrix[col]) == Series \
                else matrix[col]
            if no_of_groups >= 5:
                group_breaks = jenkspy.jenks_breaks(matrix_col, nb_class=4)
                for value in matrix[col]:
                    level = ''

                    if value >= group_breaks[4]:
                        level = 'Very High'
                        rank.append(5)
                    elif group_breaks[4] > value >= group_breaks[3]:
                        level = 'High'
                        rank.append(4)
                    elif group_breaks[3] > value >= group_breaks[2]:
                        level = 'Medium'
                        rank.append(3)
                    elif group_breaks[2] > value >= group_breaks[1]:
                        level = 'Low'
                        rank.append(2)
                    else:
                        level = 'Very Low'
                        rank.append(1)
                    level_col.append(level)
            elif 2 < no_of_groups < 5:
                group_breaks = jenkspy.jenks_breaks(matrix_col, nb_class=2)
                for value in matrix[col]:
                    level = ''
                    
                    if value >= group_breaks[2]:
                        level = 'High'
                        rank.append(3)
                    elif group_breaks[2] > value >= group_breaks[1]:
                        level = 'Medium'
                        rank.append(2)
                    else:
                        level = 'Low'
                        rank.append(1)
                    level_col.append(level)
            elif no_of_groups == 2:
                group_breaks = sorted(matrix_col)
                for value in matrix[col]:
                    level = ''
                    if value >= group_breaks[1]:
                        level = 'High'
                        rank.append(3)
                    else:
                        level = 'Low'
                        rank.append(1)
                    level_col.append(level)
            matrix[col + '_Rank'] = rank
            matrix[col + '_Level'] = level_col
        return matrix

    def _describe_cluster(self, data, col):
        clusters = data[col].unique()
        size_arr = data[col].value_counts() * 100.0 / data.shape[0]
        cluster_info = pd.DataFrame()
        for each_cluster in clusters:
            temp = data[data[col] == each_cluster].describe(
                percentiles=[.1, .25, .5, .75, .9])
            temp = temp.loc[['count', 'mean', 'std', 'min', '10%', '25%', '50%', '75%','90%', 'max']]
            temp['Cluster_Name'] = each_cluster
            cluster_info = cluster_info.append(temp)
        if col in cluster_info:
            cluster_info.drop(col, axis=1, inplace=True)
        cluster_info.set_index(['Cluster_Name', cluster_info.index], inplace=True)
        return cluster_info

    def _mad(self, data, axis=None):
        return np.median(np.abs(data - np.median(data, axis)), axis)

    def _normalizer(self):
        pass

    def _find_outliers(self, data, col, outlier_threshold=3.5):
        for field in col:
            col_data = list(data[field])
            mad_value = np.median(np.abs(col_data - np.median(col_data)))
            mad_score = np.abs(col_data - np.median(col_data)) / (
                    mad_value + 0.0001)
            data['outlier-' + field] = mad_score > outlier_threshold
        return data

    def _remove_outliers(self, data, col):
        removed = data[data[col].all(1)]
        # we can change it to any(1) in case we want to remove entire
        # row for even one true outlier.
        actual = data.drop(data[data[col].all(1)].index)
        return actual, removed

    def _get_show_columns(self):
        return self.ob_col + self.seg_col

    def _segment_data(self):
        data = self.data_frame
        data = data.groupby(self.ID).mean()
        min_max_scaler = MinMaxScaler(copy=True)
        data_scaled = min_max_scaler.fit_transform(data[self.seg_col])
        # Find most appropriate n_clusters
        if self.optimize_no_cluster=='Y':
            k_means = self._optimize_clusters(data_scaled)
            self.n_cluster = k_means.n_clusters
        else:
            k_means = KMeans(n_clusters=self.n_cluster, random_state=0)

            k_means.fit(data_scaled)
        accuracy = self.get_score(data_scaled, k_means)
        # return labelled original data for user friendly analysis
        data[self.cluster_label_column] = k_means.labels_
        #self.pickle_model(k_means)
        return data, accuracy

    def perform_outliers_detection(self, data, original_col):
        if len(self.seg_col) > 1:  # remove outliers only for more than one
            # segment column
            LOGGER.info("rid :[%s] removing outliers", self.request_id)
            data = self._find_outliers(data, self.seg_col)
            outlier_col = ['outlier-' + each for each in self.seg_col]
            data, discarded = self._remove_outliers(data, outlier_col)
            LOGGER.info("rid :[%s] percentage of data discarded : %r", self.request_id, str(
                len(discarded) / (len(data) + len(discarded) * 1.0)))
            data = data[original_col]  # Remove Outiers-ref col from data and have
            # only original fields for further processing
        else:
            LOGGER.info("rid :[%s] removing outliers for single col", self.request_id)
            data = self._find_outliers(data, self.seg_col,
                                       outlier_threshold=6.5)
            outlier_col = ['outlier-' + each for each in self.seg_col]
            data, discarded = self._remove_outliers(data, outlier_col)
            LOGGER.info("rid :[%s] percentage of data discarded :%s", self.request_id,
                        str(len(discarded) / (len(data) + len(discarded) * 1.0)))
            data = data[original_col]
        return data

    def get_score(self, data_scaled, k_means):
        accuracy = met.silhouette_score(data_scaled, k_means.labels_,
                                        sample_size=len(data_scaled) if len(data_scaled) < 2000 else 2000)
        return accuracy

    def _optimize_clusters(self, data_scaled):
        accuracy_mapping = {}
        no_of_clusters = 10
        start_cluster = 3
        for clusters in range(start_cluster, no_of_clusters + 1):
            k_means = KMeans(n_clusters=clusters, random_state=0)
            k_means.fit(data_scaled)
            accuracy = self.get_score(data_scaled, k_means)
            accuracy_mapping[accuracy] = k_means
        return k_means

    def run_segment(self):
        result, accuracy = self._segment_data()
        print ("silhouette score: ", accuracy)
        account_insights,cluster_insights, rank, cluster_size = \
            self.analyze_result(self.seg_col, result)
        return rank,account_insights, cluster_insights, cluster_size, accuracy

    def pickle_model(self, model):
        if self.pickle_client:
            self.pickle_client.pickle_model(model, self.tenant_id, self.project_id)

    def analyze_result(self, columns, result):
        column_name = columns[0]
        matrix = result.groupby(self.cluster_label_column).mean()
        cluster_size = result[self.cluster_label_column].nunique()
        matrix['Cluster_Size%'] = result.groupby(
            self.cluster_label_column).count()[column_name] / sum(
            result.groupby(self.cluster_label_column).count()[
                column_name]) * 100
        # optimize based upon % distribution of groups, 3 and 5
        # are options
        rank = self._insight(result)
        rank['Cluster_Size_Count'] = result.groupby(
            self.cluster_label_column).count()[column_name]
        rank['Cluster_Size%'] = result.groupby(
            self.cluster_label_column).count()[column_name] / sum(
            result.groupby(self.cluster_label_column).count()[
                column_name]) * 100
        show_col = ['Cluster_Size_Count', 'Cluster_Size%'] + \
                   self._get_show_columns() + [field for field in
                                               rank.columns if "_Level" in
                                               field]
        cluster_insights = (self._describe_cluster(result,
                                                   self.cluster_label_column))
        account_insights = self._get_account_insight(result, rank,
                                                     cluster_label=self.cluster_label_column).sort_values(
            self.cluster_label_column)

        cluster_insights.fillna(0, inplace=True)
        account_insights.fillna(0, inplace=True)
        return account_insights,cluster_insights, rank[show_col], cluster_size


if __name__ == '__main__':
    df=pd.read_csv('/Users/ankitgoyal/Documents/Projects/datatech/media/data_files/pageview.csv')
    print("=======")
    print(df.head())
    #data_frame, seg_col, ID=[], n_cluster=5,optimize_no_cluster='Y'):
    obj=SegmentData(df,['PAGEVIEWS','EXITS'],['AVERAGE_TIME_ON_PAGE'],'ACCOUNT_ID',3,'N')
    rank, account_insights, cluster_insights, cluster_size, accuracy =obj.run_segment()
    print ('Rank====================================','\n')
    print (rank.head())
    print("********")
    for index, row in rank.iterrows():
        print(index, row['Cluster_Size%'])
    print("********")
    print ('====================================','\n','account_insights')
    print (account_insights.head().T)
    print ('====================================','\n','cluster_insights')
    print (cluster_insights)
    print ('====================================','\n','cluster_size')
    print (cluster_size)
    print ('====================================','\n','accuracy')
    print (accuracy)