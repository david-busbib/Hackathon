import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder



class Preprocess:
    def __init__(self, input_path: str, is_test: bool, features: pd.Series = None):
        dtype_spec = {
            'arrival_time': str,
            'door_closing_time': str
        }
        try:
            self.data = pd.read_csv(input_path, encoding='cp1255', dtype=dtype_spec)
        except UnicodeDecodeError as e:
            print(f"Error reading the CSV file: {e}")
            return
        if not is_test:
            self.trip_id_unique_station = self.data["trip_id_unique_station"]
            self.labels = self.data["passengers_up"]
            self.data = self.data.drop(
                columns=['passengers_up', 'trip_id_unique_station', 'trip_id_unique', 'station_name', 'alternative',
                         'door_closing_time', 'trip_id', 'station_id'])
            self.convert_times()
            self.create_dummies_for_train()
        else:
            assert features is not None
            self.features = features
            self.trip_id_unique_station = self.data["trip_id_unique_station"]
            self.data = self.data.drop(
                columns=['trip_id_unique_station', 'trip_id_unique', 'station_name', 'alternative',
                         'door_closing_time', 'trip_id', 'station_id'])
            self.convert_times()
            self.create_dummies_for_test()

        # self.calculate_correlations()

    def plot_histograms(self):
        features = ['part', 'cluster', 'line_id']
        df = self.data.copy()

        for feature in features:
            if (feature in df.columns) and (df[feature].dtype == 'object'):
                plt.figure(figsize=(10, 6))
                sns.histplot(df[feature], bins=30, kde=False)
                plt.title(f'Histogram of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                plt.show()

    def _time_to_float(self, time_str):
        if isinstance(time_str, str):
            try:
                # Remove colons and convert to float
                digits = ''.join(time_str.split(':'))
                hours, minutes = float(digits[:2]), float(digits[2:4])
                # Calculate the total time in hours as a float
                time_in_float = hours + minutes / 60
                return time_in_float
            except ValueError:
                # Handle case where time string is not properly formatted
                return np.nan
        else:
            return np.nan

    def convert_times(self):
        for feature in ['arrival_time', 'door_closing_time']:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].apply(self._time_to_float)

    def create_dummies_for_train(self):
        df = self.data.copy()
        features_for_dummies = ['part', 'cluster', 'line_id']

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

    def create_dummies_for_test(self):
        df = self.data.copy()
        features_for_dummies = ['part', 'cluster', 'line_id']
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

    def calculate_correlations(self):
        df = self.data.copy()
        df['passengers_up'] = self.labels
        correlation_matrix = df.corr()
        sorted_correlations = correlation_matrix['passengers_up'].sort_values(ascending=False)
        print("Sorted correlations with passengers_up:")
        print(sorted_correlations)

    def get_preprocessed_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    def get_trip_id_unique_station(self):
        return self.trip_id_unique_station

    def get_features(self):
        return self.features


