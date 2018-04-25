import json
import sys

with open(sys.argv[2], 'r') as f:
    lines = f.read().split('***')

maps = []
for paragraph in lines:
    lines = paragraph.split('\n')
    map = dict([(x, y) for line in lines for x, y in line.split()])
    maps.append(map)

with open(sys.argv[1], 'r') as f:
    lines = f.read().splitlines()
assert len(lines) == len(maps), (len(lines), len(maps))
preds = []
for _line, map in zip(lines, maps):
    programs = _line.strip().replace('"', '').replace("\\n", '\n').replace('\n***\n', '').split('\n')
    for program in programs:
        processed_tokens = []
        for token in program.split():
            if token in map:
                token = map(token)
            processed_tokens.append(token)
        preds.append(' '.join(processed_tokens))
    preds.append('***')

with open(sys.argv[3], 'w') as f:
    f.write('\n'.join(preds))
