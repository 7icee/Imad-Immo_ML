# Import libraries
import pandas as pd
import numpy as np

# A class for cleaning and preprocessing a dataset
class DataCleaning:

    # Initializes the DataCleaning class with a file path and loads the data
    def __init__(self, filepath: str):
        
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

    # Drops irrelevant columns
    def drop_irrelevant(self):
        
        irrelevant_columns = [
            'house_index', 'url', 'property_subtype', 'locality', 'street', 'number',
            'box', 'furnished', 'fireplace', 'fireplaceCount', 'landSurface',
            'province', 'typeOfSale'
        ]
        self.data = self.data.drop(columns=[col for col in irrelevant_columns if col in self.data.columns])

    # Fills missing values in a specified column with zero
    def fill_empty_with_zero(self, column_name: str):

        self.data[column_name] = self.data[column_name].fillna(0)


    # Drops rows where the specified column has missing values
    def drop_rows_with_empty(self, column_name: str):

        self.data = self.data.dropna(subset=[column_name])

    # Fills missing values in a specified column with the mode   
    def fill_with_mode(self, column_name: str):

        mean_value = self.data[column_name].mode()[0]
        self.data[column_name] = self.data[column_name].fillna(mean_value)

    # Encodes the 'buildingState' column with numerical values
    def process_building_state(self):

        self.data['buildingState'] = self.data['buildingState'].fillna('UNKNOWN')
        self.data['buildingState'] = self.data['buildingState'].map({
            'GOOD': 1,
            'AS_NEW': 2,
            'TO_RENOVATE': 3,
            'TO_BE_DONE_UP': 4,
            'JUST_RENOVATED': 5,
            'TO_RESTORE': 6,
            'UNKNOWN': 0
        })
    
    # Encodes the 'property_type' column with numerical values
    def process_property_type(self):

        self.data['property_type'] = self.data['property_type'].map({
            'HOUSE': 0,
            'APARTMENT': 1
        })

    # Converts all numeric columns in the dataset to integers    
    def convert_numbers_to_int(self):

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].astype(int)

    # Performs all the cleaning and preprocessing steps
    def process(self):

        # Drop irrelevant columns
        self.drop_irrelevant()

        # Fill specified columns with zero
        columns_to_fill = ['gardenSurface', 'terraceSurface', 'surfaceOfThePlot']
        for column in columns_to_fill:
            if column in self.data.columns:
                self.fill_empty_with_zero(column)

        # Drop rows where certain columns have missing values
        rows_to_drop = ['livingArea', 'INS Code', 'Population', 'Wealth_Index', 'Density', 'price']
        for column in rows_to_drop:
            if column in self.data.columns:
                self.drop_rows_with_empty(column)
        
        # Fill missing values in 'facades' column with the mode
        self.fill_with_mode("facades")

        # Process the 'buildingState' and 'property_type' column
        self.process_building_state()
        self.process_property_type()

        # Convert numeric columns to integers
        self.convert_numbers_to_int()

        return self.data


if __name__ == "__main__":

    # Define the dataset file path and run the process
    dataset_path = "Step_2_Dataset.csv"
    processed_data = DataCleaning(dataset_path).process()

    # Save the cleaned dataset to a new CSV file.
    processed_data.to_csv("Step_3_Processed_Dataset.csv", index=False)
