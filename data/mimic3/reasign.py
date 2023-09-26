import pandas as pd

# Load the data
p0_data = pd.read_csv('p0_data.csv')
p1_data = pd.read_csv('p1_data.csv')
p2_data = pd.read_csv('p2_data.csv')
p3_data = pd.read_csv('p3_data.csv')
p4_data = pd.read_csv('p4_data.csv')

# Create a mapping from old IDs to new IDs
id_mapping = {id: f'N{str(index).zfill(5)}' for index, id in enumerate(p0_data['ID'])}

# Update the IDs in each dataframe
p0_data['ID'] = p0_data['ID'].map(id_mapping)
p1_data['ID'] = p1_data['ID'].map(id_mapping)
p2_data['ID'] = p2_data['ID'].map(id_mapping)
p3_data['ID'] = p3_data['ID'].map(id_mapping)
p4_data['ID'] = p4_data['ID'].map(id_mapping)

# Save the updated data
p0_data.to_csv('p0_data.csv', index=False)
p1_data.to_csv('p1_data.csv', index=False)
p2_data.to_csv('p2_data.csv', index=False)
p3_data.to_csv('p3_data.csv', index=False)
p4_data.to_csv('p4_data.csv', index=False)
