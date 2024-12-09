# Import libraries
import pandas as pd
import json

# Read the Excel file containing the name of the municipality and the postal code
postal_data = pd.read_excel('Step_0_Postal_Code.xlsx')

# Initialize an empty dictionary to store the link between the 'Code INS' and the postal code
code_ins_to_postal = {}

# Extract the columns 'Code INS', 'Codes postaux', and 'Langue' and store in dictionary
for index, row in postal_data.iterrows():
    code_ins = row['Code INS']
    code_postal = row['Codes postaux']
    langue = row['Langue']

    if pd.notna(langue):
        code_ins_int = int(code_ins)
        postal_list = [postal.strip() for postal in str(code_postal).split(',') if postal.strip()]
        if code_ins_int in code_ins_to_postal:
            code_ins_to_postal[code_ins_int].extend(postal_list)
        else:
            code_ins_to_postal[code_ins_int] = postal_list

# Print the resulting dictionary for verification
print(code_ins_to_postal)

# Write the dictionary in a JSON file
with open('Step_0_Code_INS_to_Postal.json', 'w') as json_file:
    json.dump(code_ins_to_postal, json_file)
