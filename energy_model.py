import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.express as px

# TODO: checking of inputs + error handling

class EnergyModel:
    """
    A class to model energy data using a RandomForestRegressor.

    Attributes:
        data_file_path (str): Path to the input CSV file.
        quantity (str): The quantity to model (e.g., 'Consumption').
        data_df (pd.DataFrame): DataFrame to store the loaded data.
        model_features (list): List of features to use in the model.
        model (RandomForestRegressor): The trained model.
        result (pd.DataFrame): DataFrame to store the results.
    """

    def __init__(self, data_file_path,quantity):
        """
        Constructor

        Args:
            data_file_path (str): Path to the input CSV file.
            quantity (str): The quantity to model (e.g., 'Consumption').
        """
        self.data_file_path = data_file_path
        self.quantity = quantity
        self.data_df = None
        self.model_features = None
        self.model = None
        self.result = pd.DataFrame()

    def load_data(self):
        """
        Load data from the CSV file into Pandas Dataframe.
        """
        self.data_df = pd.read_csv(self.data_file_path, sep=';')
        self.data_df['Time'] = pd.to_datetime(self.data_df['Time'], utc=True)

        # TODO: define verbosity parameter to mute/unmute log messages
        print('Data loaded successfully')

    def add_features(self):
        """
        Add additional modelling features to the data
        """
        # note: adding just very basic features. Many other features make sense for our problem, but lets keep it simple.
        self.data_df['day_of_year'] = self.data_df['Time'].dt.day_of_year
        self.data_df['day_of_week'] = self.data_df['Time'].dt.day_of_week
        self.data_df['is_weekend'] = self.data_df['day_of_week'] > 4 # 5 for Saturday and 6 for Sunday
        self.data_df['hours'] = self.data_df['Time'].dt.hour + self.data_df['Time'].dt.minute / 60

        self.model_features = ['day_of_year','day_of_week','hours','is_weekend']

    def clean_data(self):
        """
        Clean the data by dropping rows with missing values and ensuring complete samples per day.
        """
        # note: there are better ways handling missing variables, but lets use just deleting NaNs for the simplicity
        self.data_df.dropna(inplace=True)

        # let's use only days with complete samples. Again for the simplicity
        samples_per_day = self.data_df['day_of_year'].value_counts()
        expected_samples = 2*24 # two samples per hour
        days_to_drop = samples_per_day[samples_per_day < expected_samples].index
        self.data_df = self.data_df[~self.data_df['day_of_year'].isin(days_to_drop)]

    def prepare_data(self):
        """
        Prepare the data for modeling by splitting it into training and testing sets.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """

        # split training and evaluation data. 80% train, 20% test
        complete_days = self.data_df['day_of_year'].unique()
        split_idx = int(len(complete_days) * 0.8)
        train_days = complete_days[:split_idx]
        test_days = complete_days[split_idx:]

        data_train = self.data_df[self.data_df['day_of_year'].isin(train_days)]
        data_test = self.data_df[self.data_df['day_of_year'].isin(test_days)]
        
        self.result['Time'] = data_test['Time']

        X_train = data_train[self.model_features]
        X_test = data_test[self.model_features]
        y_train = data_train[self.quantity]
        y_test = data_test[self.quantity]

        return X_train,X_test,y_train,y_test
    
    def train_model(self,X_train,y_train):
        """
        Train the RandomForestRegressor model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        print('Fitting model...')
        self.model = RandomForestRegressor(random_state=2)
        self.model.fit(X_train,y_train)

    def evaluate_model(self,X_test,y_test):
        """
        Evaluate the trained model and store the results.

        Args:
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.

        Returns:
            float: Accuracy of the model in percentage.
        """
        print('Evaluating model...')
        y_pred = self.model.predict(X_test)
        accuracy = r2_score(y_test, y_pred) * 100 # [%]

        self.result['Measured'] = y_test
        self.result['Predicted'] = y_pred

        return accuracy
    
    def plot_results(self):
        """
        Plot the measured and predicted values using Plotly.
        """
        data_melted = self.result.melt(id_vars='Time', value_vars=self.result, var_name='Variable', value_name='Value')
        fig = px.line(data_melted, x='Time', y='Value', color='Variable', title=self.quantity, line_shape='linear')
        fig.show()
    
    def run(self):
        """
        Run the entire pipeline.
        """
        self.load_data()
        self.add_features()
        self.clean_data()
        X_train,X_test,y_train,y_test = self.prepare_data()
        self.train_model(X_train,y_train)
        accuracy = self.evaluate_model(X_test,y_test)
        self.plot_results()

        print(f'Model evaluation completed. The accuracy is {accuracy:.2f}% using the r2_score metric.')



def main():
    parser = argparse.ArgumentParser(description='Energy Model')
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--quantity', required=True, help='Quantity to model (e.g., Consumption)')
    args = parser.parse_args()

    energy_model = EnergyModel(args.input,args.quantity)
    energy_model.run()

if __name__ == '__main__':
    main()