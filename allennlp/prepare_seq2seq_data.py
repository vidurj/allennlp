import json
import random
import traceback
from collections import Counter
import spacy

from allennlp.type_checking import valid_next_characters, update_state, \
    START_SYMBOL, END_SYMBOL

nlp = spacy.load('en')
PRECISION = 7


meaningful_units = {
    # 'Dollar',
    # 'Pound',
    # 'Pound',
    # 'Hour',
    # 'Mile',
    # 'Point',
    # 'Coin',
    # 'Gallon',
    # 'Liter',
    # 'Milliliter',
    # 'Minute',
    # 'Ounce',
    # 'Foot',
    # 'Kilogram',
    # 'Ton',
    # 'Inch',
    # 'Cent',
    # 'Animal'
}

NUMBER_WORDS = {
    'twice': 2,
    'thrice': 3,
    'double': 2,
    'triple': 3,
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
        float(string.replace(',', ''))
        return True
    except:
        return False


def is_num(string):
    string = string.lower()
    if string in NUMBER_WORDS:
        return NUMBER_WORDS[string]
    try:
        temp = float(string.replace(',', ''))
        temp = round(temp, PRECISION)
        return temp
    except:
        return None


def standardize_question(text, randomize):
    source = text.replace('-', ' ')
    source_tokenized = [token.text for token in nlp(source)]
    number_to_token = {}
    tokens = ['num' + str(i) for i in range(10)]
    if randomize:
        random.shuffle(tokens)
    for index in range(len(source_tokenized)):
        number = is_num(source_tokenized[index])
        if number is not None:
            if index + 1 < len(source_tokenized) and source_tokenized[index + 1] == '%' or \
                            source_tokenized[index + 1] == 'percent':
                number /= 100.0
            number = round(number, PRECISION)
            if number not in number_to_token:
                number_to_token[number] = tokens.pop(0)
            source_tokenized[index] = number_to_token[number]
    return ' '.join(source_tokenized), number_to_token


def standardize_logical_form_with_validation(text, number_to_token, randomize):
    remaining_variable_names = ['var' + str(i) for i in range(10)]
    remaining_units = ['unit' + str(i) for i in range(2)]
    if randomize:
        random.shuffle(remaining_variable_names)
        random.shuffle(remaining_units)
    target_tokens = text.replace('?', ' ? ').replace('(', ' ( ').replace(')', ' ) ').split()
    var_assignments = {}
    type_assignments = {}
    standardized_tokens = []
    num_open_parens = 0
    num_close_parens = 0
    function_calls = []
    arg_numbers = [0]
    last_token = START_SYMBOL
    for token in target_tokens:
        valid_tokens = valid_next_characters(function_calls,
                                             arg_numbers,
                                             last_token=last_token,
                                             valid_numbers={'NUMBER'},
                                             valid_variables={'VARIABLE'},
                                             valid_types={'TYPE'})
        assert num_close_parens <= num_open_parens, (num_open_parens, num_close_parens)
        if token == '(':
            num_open_parens += 1
        elif token == ')':
            num_close_parens += 1
        elif is_strict_num(token):
            assert 'NUMBER' in valid_tokens
            number = is_num(token)
            if number in number_to_token:
                token = number_to_token[number]
            else:
                raise RuntimeError(
                    'Number {} not in number_to_token {}.'.format(number, number_to_token))
        elif token == '?':
            pass
        elif 'TYPE' in valid_tokens:
            assert 'VARIABLE' not in valid_tokens and 'NUMBER' not in valid_tokens
            if token not in type_assignments:
                assert token not in var_assignments, '{} token in var assignments too!'.format(
                    token)
                # TODO preserving types
                if token in meaningful_units:
                    type_assignments[token] = 'unit' + token
                else:
                    type_assignments[token] = remaining_units.pop(0)
            token = type_assignments[token]
        elif 'VARIABLE' in valid_tokens:
            if token not in var_assignments:
                assert token not in type_assignments, '{} token in type assignments too!'.format(token)
                var_assignments[token] = remaining_variable_names.pop(0)
            token = var_assignments[token]

        assert token.startswith('var') or token.startswith('unit') or token.startswith(
            'num') or token in valid_tokens, (token, valid_tokens, text)
        standardized_tokens.append(token)
        function_calls, arg_numbers = update_state(function_calls, arg_numbers, token)
        last_token = token
    valid_tokens = valid_next_characters(function_calls,
                                         arg_numbers,
                                         last_token=last_token,
                                         valid_numbers={'NUMBER'},
                                         valid_variables={'VARIABLE'},
                                         valid_types={'TYPE'})
    assert valid_tokens == {END_SYMBOL}, (valid_tokens, standardized_tokens, text)
    return ' '.join(standardized_tokens), (var_assignments, type_assignments)


def write_data(data, file_name, num_iters, silent=True):
    lines = []
    number_to_tokens = []
    question_numbers = []
    original_units = []
    for _ in range(num_iters):
        for question_number, question in enumerate(data):
            if question['lSemantics'] == '':
                continue
            # if question['iIndex'] != '6226':
            #     continue
            source, number_to_token = standardize_question(question['sQuestion'], randomize=True)
            try:
                target, (_, type_assignments) = standardize_logical_form_with_validation(question['lSemantics'],
                                                                     number_to_token,
                                                                     randomize=True)
                original_units.extend(type_assignments.keys())
                lines.append(source + '\t' + target)
            except:
                if not silent:
                    print(question)
                    traceback.print_exc()
                continue


    # counts = list(Counter(original_units).items())
    # counts.sort(key=lambda x: - x[1])
    # for k, v in counts:
    #     print(k, v)
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


def json_from_text_file(input_path='../euclid/data/private/rate_facts.txt', output_path='allennlp/additional_annotations.json'):
    points = []
    with open(input_path, 'r') as f:
        data = f.read().strip().splitlines()
    for line in data:
        print(line)
        question, logical_form = line.split('\t')
        points.append({'sQuestion': question.strip(), 'lSemantics': logical_form.strip()})

    with open(output_path, 'w') as f:
        json.dump(points, f)


if __name__ == '__main__':
    with open('/Users/vidurj/euclid/data/private/third_party/alg514/alg514_alignments.json',
              'r') as f:
        data = json.load(f)

    with open('allennlp/additional_annotations.json', 'r') as f:
        additional_data = json.load(f)

    with open('/Users/vidurj/euclid/data/private/train_single_sentences.txt', 'r') as f:
        train_single_sentences = json.load(f)
    print('data size', len(data))
    # write_data(data[:-100] + additional_data + train_single_sentences, 'train_abstract_types.txt', num_iters=10)
    write_data(data[-100:], 'dev.txt', num_iters=1)
    write_data(data[-100:], 'test.txt', num_iters=1)
