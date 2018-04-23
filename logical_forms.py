import json
import sys


with open(sys.argv[1], 'r') as f:
    lines = f.read().splitlines()

preds = [' '.join(json.loads(line)['predicted_tokens']) for line in lines]


with open(sys.argv[2], 'w') as f:
    f.write('\n'.join(preds))