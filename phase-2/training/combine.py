import json

file1 = "agent_trajectories_2k.json"
file2 = "test_trajectory_2k.json"
output_file = "agent_trajectories_2k_new.json"

with open(file1, "r") as f:
    data1 = json.load(f)

with open(file2, "r") as f:
    data2 = json.load(f)

if isinstance(data1, list) and isinstance(data2, list):
    combined = data1 + data2

elif isinstance(data1, dict) and isinstance(data2, dict):
    combined = {**data1, **data2}

else:
    combined = [data1, data2]

with open(output_file, "w") as f:
    json.dump(combined, f, indent=4)

print(f"Combined JSON saved to {output_file}")