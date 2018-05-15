import json
import random
import traceback
from collections import Counter, defaultdict
import spacy
import itertools

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

TIMES_WORDS = {
    'twice': ['two', 'times'],
    'thrice': ['three', 'times'],
    'double': ['two', 'times'],
    'triple': ['three', 'times'],
    'quadruple': ['four', 'times'],
    'quituple': ['five', 'times']
}

NUMBER_WORDS = {
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


def standardize_question(text, shuffle=True, is_copy=False):
    source = text.replace('-', ' ')
    temp = [token.text for token in nlp(source)]
    source_tokenized = []
    for token in temp:
        source_tokenized.extend(TIMES_WORDS.get(token.lower(), [token]))
    number_tokens = ['num' + str(i) for i in range(10)]
    if shuffle:
        random.shuffle(number_tokens)
    number_to_tokens = defaultdict(list)
    for index, token in enumerate(source_tokenized):
        number = is_num(token)
        if number is not None and (
                        source_tokenized[index + 1] == '%' or source_tokenized[
                        index + 1] == 'percent'):
            number /= 100
            number = round(number, PRECISION)
        if number is not None:
            if is_copy:
                number_to_tokens[number].append('num' + str(index))
            else:
                if number not in number_to_tokens:
                    number_to_tokens[number] = number_tokens.pop()
                source_tokenized[index] = number_to_tokens[number]
    return ' '.join(source_tokenized), number_to_tokens


def retrieve_important_numbers(text):
    source = text.replace('-', ' ')
    source_tokenized = [(token.text, token.tag_) for token in nlp(source)]
    # pos_tags = [token.tag_ for token in nlp(source)]
    print(source_tokenized)
    source_tokenized, pos_tags = zip(*source_tokenized)
    # print(pos_tags)
    important_numbers = []
    post = {'dollar', 'ticket', 'cent', 'times', 'is', 'more', 'less'}
    pre = {'is'}
    for index, token in enumerate(source_tokenized):
        number = is_num(token)
        if token.lower() == 'twice':
            important_numbers.append(2)
        elif number is None:
            continue
        elif source_tokenized[index + 1] == '%' or source_tokenized[index + 1] == 'percent':
            number /= 100
            number = round(number, PRECISION)
            important_numbers.append(number)
        elif source_tokenized[index + 1].lower() in post:
            important_numbers.append(number)
        elif source_tokenized[index - 1].lower() in pre:
            important_numbers.append(number)
        elif int(number) != number:
            important_numbers.append(number)
        elif number != 1 and number != 2 and pos_tags[index + 1] == 'NNS':
            important_numbers.append(number)
    return important_numbers


def standardize_logical_form_with_validation(text, number_to_tokens, randomize):
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
            if number in number_to_tokens:
                tokens = number_to_tokens[number]
                token = random.choice(tokens)
            else:
                raise RuntimeError(
                    'Number {} not in number_to_token {}.'.format(number, number_to_tokens))
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
                assert token not in type_assignments, '{} token in type assignments too!'.format(
                    token)
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


def prepare_synthetic_data():
    number_tokens = [str(i * 100) for i in range(1, 100)]

    def sample_question(size, objects):
        question = ['Tom', 'has']
        random.shuffle(number_tokens)
        question_index = random.randint(0, size - 1)
        for index in range(size):
            if index == question_index:
                solution = len(question)
            question.append(number_tokens[index])
            question.append(objects[index])
            question.append(',')
        question = question[:-1]
        question.extend(['.', 'How', 'many', objects[question_index], 'does', 'he', 'have', '?'])
        assert question[solution] == number_tokens[question_index]
        return ' '.join(question), 'num' + str(solution)

    with open('objects.txt', 'r') as f:
        tokens = f.read().lower().split()
    objects = []
    for token in tokens:
        if len(token) > 2:
            objects.append(token)
    print(len(objects))
    train_objects = objects[:-100]
    train = []
    questions = []
    answers = []
    # for _ in range(10000):
    #     size = random.randint(2, 5)
    #     random.shuffle(train_objects)
    #     question, answer = sample_question(size, train_objects)
    #     train.append(question + '\t' + answer)
    #     questions.append(question)
    #     answers.append(answer)
    #
    # with open('synthetic_train.txt', 'w') as f:
    #     f.write('\n'.join(train))
    #
    # with open('synthetic_train.json', 'w') as f:
    #     f.write('\n'.join(questions))
    #
    # with open('synthetic_train_solutions.txt', 'w') as f:
    #     f.write('\n'.join(answers))

    dev = []
    dev_objects = objects[-100:]
    answers = []
    questions = []
    for _ in range(1000):
        size = random.randint(6, 20)
        random.shuffle(dev_objects)
        question, answer = sample_question(size, dev_objects)
        dev.append(question + '\t' + answer)
        questions.append(json.dumps({'source': question}))
        answers.append(answer)

    with open('synthetic_dev.txt', 'w') as f:
        f.write('\n'.join(dev))

    with open('synthetic_dev.json', 'w') as f:
        f.write('\n'.join(questions))

    with open('synthetic_dev_solutions.txt', 'w') as f:
        f.write('\n'.join(answers))


def write_data(data, file_name, num_iters, randomize, is_dev=False, silent=True):
    lines = []
    number_to_tokens = []
    question_numbers = []
    original_units = []
    for _ in range(num_iters):
        for question_number, question in enumerate(data):
            if not is_dev and question['lSemantics'] == '':
                continue
            print(question['iIndex'])
            # if question['iIndex'] != '6226':
            #     continue
            source, number_to_token = standardize_question(question['sQuestion'], shuffle=not is_dev)
            try:
                if not is_dev:
                    target, (_, type_assignments) = standardize_logical_form_with_validation(
                        question['lSemantics'],
                        number_to_token,
                        randomize=randomize)
                    original_units.extend(type_assignments.keys())
                else:
                    target = 'NONE Is Dev'
                lines.append(source + '\t' + target)
                number_to_tokens.append(number_to_token)
                question_numbers.append(str(question_number))
            except:
                if is_dev or not silent:
                    print(question)
                    traceback.print_exc()
                    assert not is_dev

    assert len(number_to_tokens) == len(lines), (len(number_to_tokens), len(lines))
    # counts = list(Counter(original_units).items())
    # counts.sort(key=lambda x: - x[1])
    # for k, v in counts:
    #     print(k, v)
    print('num data points', len(lines))
    if is_dev:
        assert len(lines) == 100
    print('-' * 70)
    with open(file_name, 'w') as f:
        f.write('\n'.join(lines))
        strs = []
    for line in lines:
        strs.append(json.dumps({'source': line.split('\t')[0]}))

    with open(file_name[:-4] + '.json', 'w') as f:
        f.write('\n'.join(strs))

    with open(file_name[:-4] + '.question_numbers', 'w') as f:
        f.write('\n'.join(question_numbers))

    result_str = ''
    for number_to_token in number_to_tokens:
        for number, token in number_to_token.items():
            result_str += token + ' ' + str(number) + '\n'
        result_str += '***\n'

    with open(file_name[:-4] + '.mapping', 'w') as f:
        f.write(result_str)


def json_from_text_file(input_path='../euclid/data/private/rate_facts.txt',
                        output_path='allennlp/additional_annotations.json'):
    points = []
    with open(input_path, 'r') as f:
        data = f.read().strip().splitlines()
    for line in data:
        print(line)
        question, logical_form = line.split('\t')
        points.append({'sQuestion': question.strip(), 'lSemantics': logical_form.strip()})

    with open(output_path, 'w') as f:
        json.dump(points, f)


def create_sentence_aligned_data(alignments):
    import itertools
    with open('/Users/vidurj/euclid/data/private/third_party/alg514/kushman_annotated.json',
              'r') as f:
        data = json.load(f)

    key_to_sentence = {}
    for q in data:
        for sentence_number, sentence in enumerate(q['nlp']['sentences']):
            key = (q['iIndex'], sentence_number)
            assert key not in key_to_sentence
            key_to_sentence[key] = sentence['text']['content']

    question_to_sentence_semantics = defaultdict(lambda: [[] for _ in range(10)])
    for q in alignments:
        if 'lAlignments' in q:
            lines = q['lAlignments']
            for substring in lines:
                sentence_number, semantics = substring.split(' -> ')
                sentence_number = int(sentence_number)
                if sentence_number >= 0:
                    question_to_sentence_semantics[q['iIndex']][sentence_number].append(semantics)

    problems = []
    print(key_to_sentence.keys())
    for q_index, semantics in question_to_sentence_semantics.items():
        valid = [(i, l) for i, l in enumerate(semantics) if len(l) > 0]
        # TODO picking single sentence or all sentences
        for size in [1]:
            for sequence in itertools.combinations(valid, size):
                sequence = list(sequence)
                sequence.sort(key=lambda x: x[0])
                final_sentence = []
                final_logical_form = []
                for (i, l) in sequence:
                    final_sentence.append(key_to_sentence[(int(q_index), i)])
                    final_logical_form.extend(l)

                num_forms = len(final_logical_form)
                final_logical_form = ' '.join(final_logical_form)
                final_sentence = ' '.join(final_sentence)
                if num_forms > 1:
                    final_logical_form = '( And ' + final_logical_form + ' )'
                problem = {
                    'iIndex': q_index,
                    'sQuestion': final_sentence,
                    'lSemantics': final_logical_form
                }
                problems.append(problem)
    return problems


if __name__ == '__main__':
    with open('annotations.txt', 'r') as f:
        data = f.read().splitlines()

    logical_forms = [[]]
    for line in data:
        line = line.strip()
        if line.startswith('(') or line.startswith(')'):
            logical_forms[-1].append(line)
        else:
            print(line)
            print('*')
            logical_forms.append([])
    logical_forms = [x for x in logical_forms if len(x) > 0]
    with open('question_ids.txt', 'r') as f:
        numbers = f.read().splitlines()
    assert len(logical_forms) == len(numbers), (len(logical_forms), len(numbers))
    outputs = []
    for id, form in zip(numbers, logical_forms):
        lf = {'iIndex': int(id), 'lSemantics': ' '.join(form)}
        outputs.append(lf)
    with open('output.txt', 'w') as f:
        f.write(json.dumps(outputs))
    # prepare_synthetic_data()
    # with open('/Users/vidurj/euclid/data/private/third_party/alg514/alg514_alignments.json',
    #           'r') as f:
    #     data = json.load(f)
    #
    # with open('allennlp/additional_annotations.json', 'r') as f:
    #     additional_data = json.load(f)
    #
    # results = []
    #
    # # print(retrieve_important_numbers('The sum of 2 numbers is 15. 3 times one of the numbers is 11 less than 5 times the other. What is the smaller number? What is the larger number?'))
    #
    #
    # for index, question in enumerate(data[-100:]):
    #     print(index)
    #     print(question['iIndex'])
    #     print(question['sQuestion'])
    #     important_numbers = retrieve_important_numbers(question['sQuestion'])
    #     print(important_numbers)
    #     print('-' * 100)
    #     if len(important_numbers) == 0:
    #         results.append('*')
    #     else:
    #         results.append(' '.join([str(x) for x in set(important_numbers)]))
    #
    # with open('/Users/vidurj/euclid/data/private/dev_important_numbers.txt', 'w') as f:
    #     f.write('\n'.join(results))
    # all_train_subsets = create_sentence_aligned_data(data[:-100])
    # write_data(data[:-100], 'train.txt', randomize=True, num_iters=1)
    # write_data(data[-100:], 'dev_num.txt', is_dev=True,
    #            randomize=False, num_iters=1, silent=True)
    # write_data(data[-100:], 'test.txt', randomize=True, num_iters=1)

    # write_data(data[:3], 'train.txt', randomize=True, num_iters=500)
    # write_data(data[:3], 'dev.txt', randomize=True, num_iters=5)
