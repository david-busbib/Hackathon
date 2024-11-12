import os

import pandas as pd

from sklearn.linear_model import LinearRegression

from hackathon_code.Preprocess_2 import Preprocess2
import argparse

import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


class BasicModel:

    def __init__(self, filenameTrain, filenameTest):
        self.final_df = None
        self.dataTrain = Preprocess2(filenameTrain, False)
        self.dataTest = Preprocess2(filenameTest, True, self.dataTrain.get_features())
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=40)  # TODO switch
        # self.model =LinearRegression()
        self.y_pred = None
        self.iterpreter = False

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if not self.iterpreter:
            # Check the shape of input data
            print("Shape of input data:", X.shape)

            # Check the shape of model coefficients
            # print("Shape of model coefficients:", self.model.coef_.shape)

            try:
                y_pred = self.model.predict(X)
                y_pred = y_pred.clip(min=0)
                self.iterpreter = True
            except ValueError as e:
                print("ValueError:", e)
                print("Ensure that the input data and model coefficients have compatible shapes.")
                raise

            # df creation
            predictions_df_test = pd.DataFrame({
                'trip_id_unique': self.dataTest.get_trip_id_unique_station(),
                'arrival_time_diff_pred': y_pred
            })

            # Calculate total travel time per unique trip_id for predicted values
            total_time_pred_test = predictions_df_test.groupby('trip_id_unique')[
                'arrival_time_diff_pred'].sum().reset_index()
            total_time_pred_test.columns = ['trip_id_unique', 'trip_duration_in_minutes']

            self.final_df = total_time_pred_test

            self.y_pred = total_time_pred_test['trip_duration_in_minutes']
        return self.y_pred

    def loss(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y_pred, y)
        return mse

    def write_csv(self, filename):

        data = self.final_df
        if not os.path.exists(filename):
            # Creating a DataFrame with ID and Label columns
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)


def main(filenametest, filetrain, filenameOutput):
    modeltr = BasicModel(filetrain, filenametest)

    modeltr.fit(modeltr.dataTrain.get_preprocessed_data(), modeltr.dataTrain.get_labels())

    modeltr.predict(modeltr.dataTest.get_preprocessed_data())
    modeltr.write_csv(filenameOutput)


def Baging(filenametest, filetrain, filenameOutput):
    modeltr = BasicModel(filetrain, filenametest)
    X_train, X_test, y_train, y_test = train_test_split(
        modeltr.dataTrain.get_preprocessed_data(),
        modeltr.dataTrain.get_labels(),
        test_size=0.2,
        random_state=42
    )

    # Create and fit the Huber Regressor model
    base_estimator = DecisionTreeRegressor()  # You can use other regressors as base estimator
    bagging_model = BaggingRegressor(base_estimator=base_estimator, n_estimators=50, random_state=42)
    bagging_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = bagging_model.predict(X_train)
    y_pred_test = bagging_model.predict(X_test)

    # Calculate mean squared error
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Plot results
    plt.scatter(y_pred_test, y_test, color='blue', label='Actual', s=10)  # Smaller points
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Bagging Regression Model\nMSE (Train): {mse_train:.2f}, MSE (Test): {mse_test:.2f}')
    plt.legend()
    plt.savefig("baggingModel.png")
    plt.show()


def randomForest(filenametest, filetrain, filenameOutput):
    modeltr = BasicModel(filetrain, filenametest)

    # Define the feature to group by (e.g., 'trip_id')
    data = modeltr.dataTrain.get_preprocessed_data()

    labels = modeltr.dataTrain.get_labels()
    results = modeltr.dataTrain.get_results()
    id_s = modeltr.dataTrain.get_trip_id_unique_station()

    group_feature = 'cluster'
    if 'cluster' in data.columns:
        print("fdf")

    # # Initialize GroupShuffleSplit
    # gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Adjust test_size as needed
    #
    # # Split the data
    # train_idx, test_idx = next(gss.split(data, groups=data[group_feature]))
    #
    # # Create train and test sets
    # X_train = data.iloc[train_idx]
    # X_test = data.iloc[test_idx]
    #
    # y_test =label.iloc[test_idx]
    # y_train=label.iloc[train_idx]
    # X_train.drop(columns=[group_feature], inplace=True)
    # X_test.drop(columns=[group_feature], inplace=True)

    # train_idx, test_idx = train_test_split(
    #     data.index,
    #     test_size=0.2,
    #     random_state=42,
    # )

    group_feature = 'cluster'  # or 'trip_id_unique'

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Split the data and get indices
    train_idx, test_idx = next(gss.split(data, groups=data[group_feature]))

    data.drop(columns=[group_feature], inplace=True)

    X_train = data.loc[train_idx]
    X_test = data.loc[test_idx]

    y_train = labels.loc[train_idx]
    y_test = labels.loc[test_idx]

    results_train = results.loc[train_idx]
    results_test = results.loc[test_idx]

    id_s_train = id_s[train_idx]
    id_s_test = id_s[test_idx]

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=40)
    # rf_regressor=LinearRegression()
    # Fit the model
    rf_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred_train = rf_regressor.predict(X_train)
    y_pred_test = rf_regressor.predict(X_test)


    # df creation
    predictions_df_test = pd.DataFrame({
        'trip_id_unique': id_s_test,
        'arrival_time_diff_pred': y_pred_test
    })

    # Calculate total travel time per unique trip_id for predicted values
    total_time_pred_test = predictions_df_test.groupby('trip_id_unique')['arrival_time_diff_pred'].sum().reset_index()
    total_time_pred_test.columns = ['trip_id_unique', 'total_travel_time_pred']

    # the results

    y_reasult_test = results_test.groupby('trip_id_unique').first().reset_index()
    y_reasult_test.columns = ['trip_id_unique', 'total_travel_time_actual']
    # from here
    merged_results = pd.merge(total_time_pred_test, y_reasult_test, on='trip_id_unique', how='inner')

    # Extract the aligned predicted and actual values
    y_pred_test_total = merged_results['total_travel_time_pred']
    y_reasult_test_total = merged_results['total_travel_time_actual']

    # Calculate mean squared error
    mse_test = mean_squared_error(y_reasult_test_total, y_pred_test_total)

    # train error !!! ^^^^^ !!!!

    predictions_df_train = pd.DataFrame({
        'trip_id_unique': id_s_train,
        'arrival_time_diff_train': y_pred_train
    })

    # Calculate total travel time per unique trip_id for predicted values
    total_time_pred_train = predictions_df_train.groupby('trip_id_unique')['arrival_time_diff_train'].sum().reset_index()
    total_time_pred_train.columns = ['trip_id_unique', 'total_travel_time_train']

    # the results

    y_reasult_train = results_train.groupby('trip_id_unique').first().reset_index()
    y_reasult_train.columns = ['trip_id_unique', 'total_travel_time_actual']
    # from here
    merged_results = pd.merge(total_time_pred_train, y_reasult_train, on='trip_id_unique', how='inner')

    # Extract the aligned predicted and actual values
    y_pred_train_total = merged_results['total_travel_time_train']
    y_reasult_train_total = merged_results['total_travel_time_actual']

    # Calculate mean squared error
    mse_train = mean_squared_error(y_reasult_train_total, y_pred_train_total)

    # Calculate mean squared error
    # mse_train = mean_squared_error( , y_pred_train)
    # mse_test = mean_squared_error(y_reasult_test, y_pred_test_total)

    # Print MSE
    print(f"Train MSE: {mse_train:.2f}")
    print(f"Test MSE: {mse_test:.2f}")

    # Plotting results (optional)
    plt.scatter(y_reasult_test_total, y_pred_test_total, color='blue', label='Actual', s=10)  # Smaller points
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Random Forest Regression\nMSE (Train): {mse_train:.2f}, MSE (Test): {mse_test:.2f}')
    plt.legend()
    plt.savefig("bla_bla1.png")
    plt.show()


def baf(filenametest, filetrain, filenameOutput):
    # Define base models
    modeltr = BasicModel(filetrain, filenametest)
    X_train, X_test, y_train, y_test = train_test_split(
        modeltr.dataTrain.get_preprocessed_data(),
        modeltr.dataTrain.get_labels(),
        test_size=0.2,
        random_state=42
    )
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('bagging', BaggingRegressor(base_estimator=LinearRegression(), n_estimators=10, random_state=42))
    ]

    # Define meta-estimator (final_estimator)
    meta_estimator = LinearRegression()

    # Initialize stacking regressor
    stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_estimator)

    # Fit the stacked model
    stacked_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = stacked_model.predict(X_train)
    y_pred_test = stacked_model.predict(X_test)

    # Calculate mean squared error
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Print MSE
    print(f"Train MSE: {mse_train:.2f}")
    print(f"Test MSE: {mse_test:.2f}")

    # Plotting results (optional)
    plt.scatter(y_pred_test, y_test, color='blue', label='Actual', s=10)  # Smaller points
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Stacking with Random Forest and Bagging\nMSE (Train): {mse_train:.2f}, MSE (Test): {mse_test:.2f}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument('--training_set', type=str, required=True, help='Path to the training set')
    parser.add_argument('--test_set', type=str, required=True, help='Path to the test set')
    parser.add_argument('--out', type=str, required=True, help='Output path')

    args = parser.parse_args()
    print(args)

    main(args.test_set,args.training_set,  args.out)
    # randomForest(args.test_set, args.training_set, args.out)

"""
 --training_set /cs/usr/dbusbib123/PycharmProjects/pythonProject4/train_bus_schedule.csv
--test_set /cs/usr/dbusbib123/PycharmProjects/pythonProject4/X_passengers_up.csv
--out /cs/usr/dbusbib123/PycharmProjects/pythonProject4/.csv"""
