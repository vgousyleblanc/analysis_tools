import json

# Load the JSON config file
with open('/eos/user/v/vgousyle/proton_search/TriggerConfig/configurations/lep_v51_config.json', 'r') as f:
    config = json.load(f)

# Extract ACT channel mappings (channels 12-23)
channel_mapping = {}
input_signals = config['input_signals']

for channel_id in range(12, 24):
    channel_str = str(channel_id)
    if channel_str in input_signals:
        short_name = input_signals[channel_str]['short_name']
        channel_mapping[channel_id] = short_name

# Generate the Python code
print("self.channel_mapping = {", end="")
items = [f'{k}: "{v}"' for k, v in sorted(channel_mapping.items())]
print(", ".join(items))
print("}")

# Also print as a pretty Python dict for reference
print("\n# Or in pretty format:")
print("self.channel_mapping = {")
for channel_id, short_name in sorted(channel_mapping.items()):
    print(f"    {channel_id}: \"{short_name}\",")
print("}")
