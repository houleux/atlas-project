import pandas as pd
from unidecode import unidecode
import matplotlib.pyplot as plt
import numpy as np


cot = pd.read_csv('cities.csv')
countries = pd.read_csv('country.csv')

cities = cot.loc[:, ['city']]
cities.rename(columns={'city':'place'}, inplace=True)
cities['place'] = cities['place'].str.strip().str.lower().apply(unidecode)
cities['Starting letter'] = cities['place'].str[0]
cities['Ending letter'] = cities['place'].str[-1]

countries.rename(columns={'value': 'place'}, inplace=True)
countries['place'] = countries['place'].str.strip().str.lower().apply(unidecode)
countries['Starting letter'] = countries['place'].str[0]
countries['Ending letter'] = countries['place'].str[-1]

countries.drop('id', axis=1, inplace=True)

places = pd.merge(cities.head(500), countries, on=['place', 'Starting letter', 'Ending letter'], how='outer')


start_count = places['Starting letter'].value_counts().reset_index()
start_count.columns = ['Letter', 'Start Count']

end_count = places['Ending letter'].value_counts().reset_index()
end_count.columns = ['Letter', 'End Count']

letter_counts = pd.merge(start_count, end_count, on='Letter', how='outer').fillna(0)

letter_counts = letter_counts.sort_values('Letter').reset_index(drop=True)
letter_counts = letter_counts.drop(0)


letter_counts['disparity'] = letter_counts['Start Count'] - letter_counts['End Count']

plt.figure(figsize=(12, 6))

# Bar width for grouping the bars side by side
bar_width = 0.35
index = np.arange(len(letter_counts['Letter']))

# Create the bar plot
plt.bar(index, letter_counts['Start Count'], bar_width, label='Start Count', color='skyblue')
plt.bar(index + bar_width, letter_counts['End Count'], bar_width, label='End Count', color='salmon')

# Label and title
plt.xlabel('Letters')
plt.ylabel('Counts')
plt.title('Start vs End Letter Counts for places Names')
plt.xticks(index + bar_width / 2, letter_counts['Letter'])

# Show legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

print(letter_counts.sort_values('Start Count', ascending=False))