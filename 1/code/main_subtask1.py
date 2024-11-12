import os

import pandas as pd

from sklearn.linear_model import LinearRegression

from hackathon_code.Preprocess import Preprocess
import argparse


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

def main(training_set, test_set, out):
    # Your code here
    print(f"Training set path: {training_set}")
    print(f"Test set path: {test_set}")
    print(f"Output path: {out}")
class BasicModel:

    def __init__(self,filenameTrain,filenameTest):
        self.dataTrain= Preprocess(filenameTrain,False)
        self.dataTest = Preprocess(filenameTest,True,self.dataTrain.get_features())

        self.model=  RandomForestRegressor(n_estimators=100, random_state=42,max_depth=40)
        # self.model =LinearRegression()
        self.y_pred=None
        self.iterpreter= False

        pass


    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self, X):
        if not self.iterpreter:
            # Check the shape of input data
            print("Shape of input data:", X.shape)

            # Check the shape of model coefficients
            # print("Shape of model coefficients:", self.model.coef_.shape)

            try:
                y_pred = self.model.predict(X)
                self.y_pred = y_pred.clip(min=0)
                self.iterpreter = True
            except ValueError as e:
                print("ValueError:", e)
                print("Ensure that the input data and model coefficients have compatible shapes.")
                raise

        return self.y_pred

    def loss(self,X,y):
        y_pred =self.predict(X)
        mse=mean_squared_error(y_pred,
y)

        return mse
    def write_csv(self,filename):

        data = {
            'trip_id_unique_station': self.dataTest.get_trip_id_unique_station(),
            'passengers_up': self.y_pred
        }
        if not os.path.exists(filename):
            # Creating a DataFrame with ID and Label columns
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)

def main(filenametest,filetrain,filenameOutput):
    modeltr =BasicModel(filetrain,filenametest)

    modeltr.fit(modeltr.dataTrain.get_preprocessed_data(),modeltr.dataTrain.get_labels())

    modeltr.predict(modeltr.dataTest.get_preprocessed_data())
    modeltr.write_csv(filenameOutput)



def Baging(filenametest,filetrain,filenameOutput):
    modeltr =BasicModel(filetrain,filenametest)

    X_train, X_test, y_train, y_test = train_test_split(
        modeltr.dataTrain.get_preprocessed_data(), modeltr.dataTrain.get_labels(),
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
    X, y = modeltr.dataTrain.get_preprocessed_data(), modeltr.dataTrain.get_labels()

    percentages = np.linspace(0.4, 1.0, 10)
    train_mses = []
    test_mses = []

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42,
                                         max_depth=40)
    rf_regressor.fit(X_train, y_train)

    y_pred_train = rf_regressor.predict(X_train)
    y_pred_test = rf_regressor.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    train_mses.append(mse_train)
    test_mses.append(mse_test)
    plt.scatter(y_pred_test, y_test, color='blue', label='Actual', s=10)  # Smaller points
    plt.xlabel('y_pred')
    plt.ylabel('y_test')
    plt.title(f'Random Forest Regression Performance with Mse={mse_test} ')
    plt.legend()
    plt.grid(True)
    plt.show()


# Assuming BasicModel and other necessary components are defined elsewhere in your code
def baf(filenametest,filetrain,filenameOutput):
    # Define base models
    modeltr = BasicModel(filetrain, filenametest)
    X_train, X_test, y_train, y_test = train_test_split(
        modeltr.dataTrain.get_preprocessed_data(), modeltr.dataTrain.get_labels(),
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
    stacked_model = StackingRegressor(estimators=
base_models, final_estimator=meta_estimator)


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

    main(args.test_set,args.training_set,  args.out)