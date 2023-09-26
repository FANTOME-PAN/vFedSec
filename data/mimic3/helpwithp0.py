import csv

# Input file
input_file = 'p0_data.csv'

# Output file
output_file = 'p0_data_modified.csv'

# Add 'filler' column with value 0 to the CSV file
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Modify the data by adding the 'filler' column with 0 values
for row in data:
    row.append('0')

# Write the modified data to the output file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"New column 'filler' with values 0 added to {output_file}")

