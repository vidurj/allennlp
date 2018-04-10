import json
import pickle

import spacy
import random

nlp = spacy.load('en')



NUMBER_WORDS = {
    'twice': 2,
    'thrice': 3,
    'quadruple': 4,
    'quituple': 5,
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
    'eighth': 8,
    'ninth': 9,
    'tenth': 10,
    'hundred': 100,
    'thousand': 1000,
    'million': 1000000
}


def is_strict_num(string):
    try:
        float(string)
        return True
    except:
        return False


def is_num(string):
    string = string.lower()
    if string in NUMBER_WORDS:
        return NUMBER_WORDS[string]
    try:
        temp = float(string.replace(',', ''))
        temp = int(temp * 10000) / 10000.0
        return temp
    except:
        return None


def write_data(data, file_name, num_iters):
    lines = []
    max_len = 0
    number_to_tokens = []
    question_numbers = []
    for _ in range(num_iters):
        for question_number, question in enumerate(data):
            source = question['sQuestion'].replace('-', ' ')
            source_tokenized = [token.text for token in nlp(source)]
            number_to_token = {}
            tokens = ['num' + str(i) for i in range(10)]
            for index in range(len(source_tokenized)):
                number = is_num(source_tokenized[index])
                if number is not None:
                    if source_tokenized[index + 1] == '%' or source_tokenized[index + 1] == 'percent':
                        number /= 100.0
                    if number not in number_to_token:
                        new_var = random.choice(tokens)
                        number_to_token[number] = new_var
                        tokens.remove(new_var)
                    source_tokenized[index] = number_to_token[number]

            target = question['lSemantics']
            target_tokens = target.replace('(', ' ( ').replace(')', ' ) ').split()
            if len(target_tokens) == 0:
                continue
            assignments = {}
            remaining_variable_names = ['var' + str(i) for i in range(10)]
            remaining_units = ['unit' + str(i) for i in range(2)]
            take = True
            for index in range(1, len(target_tokens)):
                token = target_tokens[index]
                number = is_num(token)
                if is_strict_num(token) and number in number_to_token:
                    target_tokens[index] = number_to_token[number]
                elif target_tokens[index - 1] == '(' or token == '(' or token == ')':
                    continue
                else:
                    if is_strict_num(token):
                        print(token, question['iIndex'], source_tokenized)
                        take = False
                        break
                    if token.islower():
                        if token not in assignments:
                            new_var = random.choice(remaining_variable_names)
                            remaining_variable_names.remove(new_var)
                            if token[0] == '?':
                                new_var = '? ' + new_var
                            assignments[target_tokens[index]] = new_var
                        target_tokens[index] = assignments[target_tokens[index]]
                    else:
                        if token not in assignments:
                            new_var = random.choice(remaining_units)
                            remaining_units.remove(new_var)
                            assignments[target_tokens[index]] = new_var
                        target_tokens[index] = assignments[target_tokens[index]]
            if take:
                max_len = max(max_len, len(target_tokens))
                lines.append(' '.join(source_tokenized) + '\t' + ' '.join(target_tokens))
                number_to_tokens.append(number_to_token)
                question_numbers.append(str(question_number))

    print('max length', max_len)
    print('num data points', len(lines))
    print('-' * 70)
    with open(file_name, 'w') as f:
        f.write('\n'.join(lines))

        strs = []
    for line in lines:
        strs.append(json.dumps({'source': line.split('\t')[0]}))

    with open(file_name[:-4] + '.json', 'w') as f:
        f.write('\n'.join(strs))

    result_str = ''
    for number_to_token in number_to_tokens:
        for number, token in number_to_token.items():
            result_str += token + ' ' + str(number) + '\n'
        result_str += '***\n'

    with open(file_name[:-4] + '.mapping', 'w') as f:
        f.write(result_str)

    with open(file_name[:-4] + '.question_numbers', 'w') as f:
        f.write('\n'.join(question_numbers))


if __name__ == '__main__':

    # with open('/Users/vidurj/euclid/data/private/single_sentences_dev.txt', 'r') as f:
    #     data = json.load(f)
    #
    # write_data(data, 'single_sentences_dev.txt', num_iters=2)

    with open('/Users/vidurj/euclid/data/private/third_party/alg514/alg514_alignments.json', 'r') as f:
        data = json.load(f)
    print('data size', len(data))
    write_data(data[:-100], 'train.txt', num_iters=3)
    write_data(data[-100:], 'dev.txt', num_iters=1)
    write_data(data[-100:], 'test.txt', num_iters=1)
