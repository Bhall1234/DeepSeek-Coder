import json

# Step 1: Read JSON File
with open('incorrect_responses_evolution_1.json', 'r') as file:
    data = json.load(file)

# Step 2: Modify Content
formatted_data = []
for entry in data:
    formatted_entry = {
        "instruction": entry["instruction"],
        "input": "",
        "output": entry["output"]
    }
    formatted_data.append(formatted_entry)

# Step 3: Write Modified Content
with open('formatted_dataset.json', 'w') as file:
    json.dump(formatted_data, file, indent=4)