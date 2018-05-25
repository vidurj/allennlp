import json
import sys



with open(sys.argv[1], 'r') as f:
    lines = f.read().splitlines()

with open(sys.argv[2], 'r') as f:
    _lines = f.read().strip().splitlines()
    maps = [dict()]
    for line in _lines:
        if line.strip() == '***':
            maps.append(dict())
        else:
            key, value = line.split()
            maps[-1][key] = value
    maps = maps[:-1]
    assert len(lines) == len(maps), (len(lines), len(maps))


preds = []
for line, map in zip(lines, maps):
    processed_tokens = []
    for token in json.loads(line)['predicted_tokens']:
        if token.startswith('num'):
            if token in map:
                token = map[token]
            else:
                print('token {} not in map!'.format(token))
        processed_tokens.append(token)
    preds.append(' '.join(processed_tokens))

with open(sys.argv[3], 'w') as f:
    f.write('\n'.join(preds))
