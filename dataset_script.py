import csv
import random

TARGET_DATASET = "cyberbullying_binary_dataset.csv"

# Open files for each dataset
f_aggression = open('aggression_parsed_dataset.csv', encoding='utf-8')
f_toxicity = open('toxicity_parsed_dataset.csv', encoding='utf-8')
f_racism = open('twitter_racism_parsed_dataset.csv', encoding='utf-8')

# Use csv readers for datasets
csv_aggression = csv.reader(f_aggression)
csv_toxicity = csv.reader(f_toxicity)
csv_racism = csv.reader(f_racism)

# Create file for combined data
f_combined = open(TARGET_DATASET, mode="w+", encoding='utf-8', newline='')
csv_writer = csv.writer(f_combined)

# Extract data from datasets
neutral = []

def parse_dataset(dataset, text_index, target_index, output_index):
    parsed_dataset = []

    index = 0
    for row in dataset:
        # Skip first row
        if (index == 0):
            index += 1
            continue
        
        # Extract text
        if (not row[text_index]):
            continue

        parsed_row = [row[text_index], '0', '0']

        # Put target into correct output column
        if (float(row[target_index]) > 0.5):
            parsed_row[output_index] = 1

        # Extract neutral data
        else:
            parsed_row[1] = 1
            neutral.append(parsed_row)
            continue
        
        # Append to result
        parsed_dataset.append(parsed_row)
        index += 1
    
    return parsed_dataset

aggression = parse_dataset(csv_aggression, 1, 3, 2)
toxicity = parse_dataset(csv_toxicity, 1, 3, 2)
racism = parse_dataset(csv_racism, 2, 4, 2)

cyberbullying = []
cyberbullying.extend(aggression)
cyberbullying.extend(toxicity)
cyberbullying.extend(racism)

min_length = min(len(cyberbullying), len(neutral))

# Balance data
neutral = random.sample(neutral, min_length)
cyberbullying = random.sample(cyberbullying, min_length)

# Combine datasets
combined_dataset = []

combined_dataset.append(['text', 'neutral', 'cyberbullying'])
combined_dataset.extend(neutral)
combined_dataset.extend(cyberbullying)

# # Write header row
print(f"Writing new dataset to {TARGET_DATASET}")
csv_writer.writerows(combined_dataset)

# Close files gracefully
f_combined.close()
f_aggression.close()
f_toxicity.close()
f_racism.close()