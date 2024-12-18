# Import libraries
import pandas as pd
import json

# Read the property_data.csv and postal_data.xlsx
property_data = pd.read_csv('Step_1_Pre_Dataset.csv')
postal_data = pd.read_excel('Step_1_Postal_Data.xlsx')

# Convert postal_data.xlsx into a dictionary with 'Code_INS' as the key
postal_data_dict = postal_data.set_index('Code_INS')[['Population', 'Wealth_Index', 'Density']].to_dict(orient='index')

# Load the mapping of Code_INS to postal codes from a JSON file
with open('Step_0_Code_INS_to_Postal.json', 'r') as json_file:
    ins_to_postal = json.load(json_file)

# Define a function to link the 'Code INS' based on the postal codes
def get_ins_code(postal_code):
    for ins_code, postal_list in ins_to_postal.items():
        if str(int(postal_code)) in postal_list:
            return int(ins_code)
    return None

# Define a function to get the population for a given INS code
def get_population(ins_code):
    if ins_code in postal_data_dict:
        return postal_data_dict[ins_code]['Population']
    return None

# Define a function to get the wealth index for a given INS code
def get_wealth_index(ins_code):
    if ins_code in postal_data_dict:
        return postal_data_dict[ins_code]['Wealth_Index']
    return None

# Define a function to get the density for a given INS code
def get_density(ins_code):
    if ins_code in postal_data_dict:
        return postal_data_dict[ins_code]['Density']
    return None

# Add a new column "INS_Code" to the property data by applying the 'get_ins_code' function to the 'postal_code' column
property_data['INS_Code'] = property_data['postal_code'].apply(get_ins_code)

# Add columns for Population, Wealth Index, and Density by applying the corresponding functions to the 'INS_Code' column
property_data['Population'] = property_data['INS_Code'].apply(get_population)
property_data['Wealth_Index'] = property_data['INS_Code'].apply(get_wealth_index)
property_data['Density'] = property_data['INS_Code'].apply(get_density)

# Save the updated property data to a new CSV file
property_data.to_csv('Step_2_Dataset.csv', index=False)
