import pandas as pd
import json

# Load Excel file
excel_file_path = 'Step_1_Postal_Data.xlsx'
df = pd.read_excel(excel_file_path)

# Load JSON
json_file_path = 'Step_0_Code_INS_to_Postal.json'
with open(json_file_path, 'r') as json_file:
    postal_code_dict = json.load(json_file)
    postal_code_dict = {int(key): value for key, value in postal_code_dict.items()}

# Convert 'Code_INS' to handle ','
df['Code_INS'] = df['Code_INS'].astype(str).str.replace(',', '').astype(int)

# Create a column for Postal_code based on the Code_INS
df['Postal_code'] = df['Code_INS'].apply(lambda ins_code: postal_code_dict[ins_code][0] if ins_code in postal_code_dict else None)

# Save the result  to a new Excel file
output_file_path = 'updated_file_with_postal_codes.xlsx'
df.to_excel(output_file_path, index=False)
