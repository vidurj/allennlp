import json
import sys


with open(sys.argv[1], 'r') as f:
    lines = f.read().splitlines()

all_preds = []
for line in lines:
    preds = ' '.join(json.loads(line)['predicted_tokens'])
    all_preds.append(preds)

with open(sys.argv[2], 'w') as f:
    f.write('\n'.join(all_preds))