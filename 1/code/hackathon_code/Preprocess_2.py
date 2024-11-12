import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



class Preprocess2:
    def __init__(self, input_path: str, is_test: bool, features: pd.Series = None):
        dtype_spec = {
            'arrival_time': str,
            'door_closing_time': str
        }
        self.is_test = is_test

        try:
            self.data = pd.read_csv(input_path, encoding='cp1255', dtype=dtype_spec)
        except UnicodeDecodeError as e:
            print(f"Error reading the CSV file: {e}")
            return
        if not is_test:
            self.trip_id_unique_ = self.data["trip_id_unique"]
            self.labels = None
            self.results = None
            self.data = self.data.drop(
                columns=['trip_id_unique_station', 'station_name', 'alternative',
                         'door_closing_time', 'station_id', 'trip_id'])
            self.convert_times()
            self.create_dummies_for_train()
            self.add_diffs()
        else:
            assert features is not None
            print("trip_id_unique" in self.data.columns )
            self.features = features
            self.trip_id_unique_ = self.data["trip_id_unique"]
            self.data = self.data.drop(
                columns=['trip_id_unique_station', 'station_name', 'alternative',
                         'door_closing_time', 'station_id', 'trip_id'])
            self.convert_times()
            self.add_diffs()
            self.create_dummies_for_test()



    def _time_to_float(self, time_str):
        if isinstance(time_str, str):
            try:
                # Remove colons and convert to float
                digits = ''.join(time_str.split(':'))
                hours, minutes, seconds = float(digits[:2]), float(digits[2:4]), float(digits[4:])
                # Calculate the total time in hours as a float
                time_in_float = (hours * 60) + minutes + (seconds / 60)
                return time_in_float
            except ValueError:
                # Handle case where time string is not properly formatted
                return np.nan
        else:
            return np.nan

    def convert_times(self):
        self.data["arrival_time"] = self.data["arrival_time"].apply(self._time_to_float)

    def create_dummies_for_train(self):
        df = self.data.copy()
        features_for_dummies = ['part', 'cluster', 'line_id'] # todo cluster

        self.frequent_categories = {}
        for feature in features_for_dummies:
            if feature in df.columns:
                # Get the frequency of each category
                frequencies = df[feature].value_counts()
                # Determine the least frequent categories
                least_frequent = frequencies[frequencies < frequencies.quantile(0.05)].index
                # Store the frequent categories for use with test data
                self.frequent_categories[feature] = set(frequencies.index) - set(least_frequent)
                # Replace least frequent categories with 'unknown'
                df[feature] = df[feature].apply(lambda x: x if x in self.frequent_categories[feature] else 'unknown')

        # OneHotEncode the modified features
        df = pd.get_dummies(df, columns=features_for_dummies)

        # Ensure dummy variables are integers (0 and 1)
        dummy_columns = df.columns[df.columns.str.startswith(tuple(features_for_dummies))]
        df[dummy_columns] = df[dummy_columns].astype(int)

        encoder = LabelEncoder()
        if 'arrival_is_estimated' in df.columns:
            df['arrival_is_estimated'] = encoder.fit_transform(df['arrival_is_estimated'])

        self.data = df
        self.features = df.columns

    def add_diffs(self):
        self.data = self.data.sort_values(by=['trip_id_unique', 'direction', 'station_index'])
        for feature in ['station_index', 'arrival_time', 'longitude', 'latitude']:
            # Sort the dataframe
            if self.is_test and feature == 'arrival_time':
                continue
            new_feature = feature + '_diff'
            self.data[new_feature] = self.data[feature].diff()

            if feature == 'arrival_time':
                self.data[new_feature] = self.data[new_feature].fillna(0).apply(lambda x: x if x >= - 1000 else x + 24 * 60)
                self.data[new_feature] = self.data[new_feature].fillna(0).apply(lambda x: max(x, 0))
                self.data[new_feature] = self.data[new_feature].fillna(0).apply(lambda x: x if x <= 30 else 0)
            elif feature =="station_index":
                self.data[new_feature] = self.data[new_feature].fillna(0).apply(lambda x: max(x, 0))
            elif feature != 'arrival_time':
                self.data[new_feature] = self.data[new_feature].fillna(0).apply(lambda x: abs(x))

            cols = self.data.columns.tolist()
            new_feature_idx = cols.index(new_feature)

            # Create the new order of columns
            new_cols = cols[:new_feature_idx] + [new_feature] + cols[new_feature_idx + 1:]
            self.data = self.data[new_cols]
            self.data.loc[self.data['station_index_diff'] == 0, new_feature] = 0

        # Drop the 'arrival_time' column
        if not self.is_test:
            total_travel_time = self.data.groupby('trip_id_unique')['arrival_time_diff'].sum().reset_index()
            total_travel_time.columns = ['trip_id_unique', 'total_travel_time']
            self.data = self.data.merge(total_travel_time, on='trip_id_unique')
            self.results = self.data[['trip_id_unique', 'total_travel_time']]

            self.data['arrival_time_diff'] = self.data['arrival_time_diff'].fillna(0).apply(lambda x: x if x <= 4 * 60 else 0)

            self.labels = self.data['arrival_time_diff']
            self.data = self.data.drop(columns=['trip_id_unique' , 'total_travel_time','arrival_time_diff','arrival_time','longitude','latitude','station_index_diff'])
        else:
            self.data = self.data.drop(columns=[ 'arrival_time','longitude','latitude','station_index_diff'])


    # def calculate_correlations(self):
    #     df = self.data.copy()
    #     correlation_matrix = df.corr()
    #     sorted_correlations = correlation_matrix['arrival_time_diff'].sort_values(ascending=False)
    #
    #     print("Sorted correlations with arrival_time_diff:")
    #     print(sorted_correlations)
    #
    #     print()
    def create_dummies_for_test(self):
        df = self.data.copy()
        features_for_dummies = ['part', 'cluster', 'line_id']  # todo ommetedd cluster
        # Create dummy variables
        df = pd.get_dummies(df, columns=features_for_dummies)

        # Ensure dummy variables are integers (0 and 1)
        dummy_columns = df.columns[df.columns.str.startswith(tuple(features_for_dummies))]
        df[dummy_columns] = df[dummy_columns].astype(int)
        features = self.features.tolist()
        for prefix in features_for_dummies:
            for feature in df.columns:
                if feature.startswith(prefix):
                    if feature not in features:
                        unknown_feature = prefix + "_unknown"
                        if unknown_feature in df.columns:
                            df[unknown_feature] += df[feature]
                            df.drop(columns=[feature])
                        else:
                            df = df.rename(columns={feature: unknown_feature})
        train_set_features = set(features)
        test_set_features = set(df.columns)
        not_in_test = train_set_features - test_set_features
        for feature in not_in_test:
            df[feature] = np.zeros(len(df))
        df = df.reindex(columns=self.features)

        # Encode categorical columns if needed
        encoder = LabelEncoder()
        if 'arrival_is_estimated' in df.columns:
            df['arrival_is_estimated'] = encoder.fit_transform(df['arrival_is_estimated'])

        self.data = df
        self.test_features = df.columns

    def get_preprocessed_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    def get_results(self):
        return self.results

    def get_trip_id_unique_station(self):
        return self.trip_id_unique_

    def get_features(self):
        return self.data.columns

