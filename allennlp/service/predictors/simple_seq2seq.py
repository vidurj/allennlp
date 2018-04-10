from overrides import overrides
import spacy
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor
from typing import List, Tuple
import json
from allennlp.prepare_seq2seq_data import is_num
import random
from allennlp.data.dataset import Batch
from allennlp.common import Registrable
from allennlp.common.util import JsonDict, sanitize
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.common import Params
import sys
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, Set
from allennlp.models.archival import load_archive
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.training.trainer import Trainer
from allennlp.commands.train import datasets_from_params
import torch
from allennlp.models.model import Model
from allennlp.training.optimizers import Optimizer
from allennlp.common.tqdm import Tqdm
import numpy as np
import os
import subprocess
import cmd
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL


def instances_to_batch(instances, model, for_training, cuda_device=0):
    batch = Batch(instances)
    batch.index_instances(model.vocab)
    padding_lengths = batch.get_padding_lengths()
    return batch.as_tensor_dict(padding_lengths,
                                cuda_device=cuda_device,
                                for_training=for_training)


class SimpleTrainer:
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 train_dataset: Iterable[Instance],
                 iterator: DataIterator,
                 cuda_device=0
                 ):
        self._optimizer = optimizer
        self._model = model
        self._train_data = train_dataset
        self._iterator = iterator
        self.cuda_device = cuda_device
        self.cur_position = len(self._train_data)
        self.batch_size = 20

    def _create_batch(self, new_instance, new_instances):
        if self.cur_position + self.batch_size >= len(self._train_data):
            random.shuffle(self._train_data)
            self.cur_position = 0
        instances = self._train_data[self.cur_position: self.cur_position + self.batch_size]
        instances.extend(new_instances)
        instances.append(new_instance)
        return instances_to_batch(instances,
                                  self._model,
                                  for_training=True,
                                  cuda_device=self.cuda_device)

    def train(self, new_instance, new_instances):
        flag = True
        gold_prediction = new_instance.fields['target_tokens']
        gold_prediction.index(self._model.vocab)
        gold_prediction = gold_prediction._indexed_tokens['tokens'][1:]
        gold_length = len(gold_prediction)
        count = 0
        while flag:
            self._optimizer.zero_grad()
            batch = self._create_batch(new_instance, new_instances)
            output_dict = self._model(**batch)
            loss = output_dict["loss"]
            loss += self._model.get_regularization_penalty()
            loss.backward()
            self._optimizer.step()
            prediction = output_dict['predictions'][-1].data.cpu().numpy()
            if np.all(prediction[:gold_length] == gold_prediction):
                count += 1
            else:
                count = 0
            if count > 5:
                break


@Predictor.register('simple_seq2seq_beam')
class SimpleSeq2SeqPredictorBeam(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source), {}

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)

        dataset = Batch([instance])
        dataset.index_instances(self._model.vocab)
        model_input = dataset.as_tensor_dict(cuda_device=cuda_device, for_training=False)
        output = self._model.beam_search(model_input['source_tokens'], bestk=100)
        return output

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict], cuda_device: int = -1) -> List[JsonDict]:
        instances, return_dicts = zip(*self._batch_json_to_instances(inputs))
        output_string = ''
        for instance in instances:
            instance_output = self._model.beam_search(instance.fields['source_tokens'].as_array(),
                                                      bestk=100)
            output_string += instance_output
        return output_string


def clean_text(text):
    nlp = spacy.load('en')
    source = text.replace('-', ' ')
    source_tokenized = [token.text for token in nlp(source)]
    number_to_token = {}
    tokens = ['num' + str(i) for i in range(10)]
    for index in range(len(source_tokenized)):
        number = is_num(source_tokenized[index])
        if number is not None:
            if index + 1 < len(source_tokenized) and source_tokenized[index + 1] == '%' or \
                            source_tokenized[index + 1] == 'percent':
                number /= 100.0
            if number not in number_to_token:
                new_var = tokens[0]
                tokens = tokens[1:]
                number_to_token[number] = new_var
            source_tokenized[index] = number_to_token[number]
    return ' '.join(source_tokenized)


def instance_to_source_string(instance):
    tokens = [token.text for token in instance.fields['source_tokens'].tokens]
    assert tokens[0] == START_SYMBOL and tokens[-1] == END_SYMBOL, (tokens[0], tokens[-1])
    return ' '.join(tokens[1:-1])


def instance_to_target_string(instance):
    tokens = [token.text for token in instance.fields['target_tokens'].tokens]
    assert tokens[0] == START_SYMBOL and tokens[-1] == END_SYMBOL, (tokens[0], tokens[-1])
    return ' '.join(tokens[1:-1])


class Interpreter(cmd.Cmd):
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = '(teacher)'
    file = None

    def __init__(self, model, dataset_reader, trainer):
        super().__init__()
        self._model = model
        self._dataset_reader = dataset_reader
        self._trainer = trainer
        self.new_instances = []
        self.serialization_dir = 'retrained'
        self.last_labeled_instance = None

    def do_last_instance(self, _):
        source = instance_to_source_string(self.last_labeled_instance)
        target = instance_to_target_string(self.last_labeled_instance)
        print('source: {0}\ntarget: {1}'.format(source, target))

    def do_add(self, _):
        if self.last_labeled_instance is None:
            print('No unadded labeled instance exists.')
        else:
            self.new_instances.append(self.last_labeled_instance)
            self.last_labeled_instance = None

    def do_solve(self, text):
        source = clean_text(text.strip())
        instance = self._dataset_reader.text_to_instance(source)
        batch = instances_to_batch([instance], self._model, for_training=False)
        predictions = self._model.beam_search(batch['source_tokens'], bestk=1)
        target = predictions.split('\n')[0]
        print(target)
        self.last_labeled_instance = self._dataset_reader.text_to_instance(source, target)

    def do_add_instance(self, line):
        source, target = line.split('\t')
        labeled_instance = self._dataset_reader.text_to_instance(source, target)
        self._trainer.train(labeled_instance, self.new_instances)
        self.new_instances.append(labeled_instance)

    def do_learn(self, text):
        if self.last_labeled_instance is None:
            print('No unadded labeled instance exists.')
        else:
            source = instance_to_source_string(self.last_labeled_instance)
            target = ' '.join(
                text.strip().replace('(', ' ( ').replace(')', ' ) ').replace('?', ' ? ')
                .split())
            labeled_instance = self._dataset_reader.text_to_instance(source, target)
            self._trainer.train(labeled_instance, self.new_instances)
            self.new_instances.append(labeled_instance)

    def do_add_from_file(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f.read().splitlines():
                    self.do_add_instance(line)
        else:
            print('{} path does not exists'.format(file_path))

    def do_save(self):
        source_tokens = [instance_to_source_string(instance) for instance in self.new_instances]
        target_tokens = [instance_to_target_string(instance) for instance in self.new_instances]
        annotations = [source + '\t' + target for source, target in
                       zip(source_tokens, target_tokens)]
        with open('annotations.txt', 'w') as f:
            f.write('\n'.join(annotations))
        model_path = os.path.join(self.serialization_dir, "model_state.th")
        model_state = self._model.state_dict()
        torch.save(model_state, model_path)

    def complete_solve(self, text, line, begidx, endidx):
        pass


@Predictor.register('simple_seq2seq_interactive')
class SimpleSeq2SeqPredictorInteractive(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source), {}

    @overrides
    def predict_json(self, _: JsonDict, cuda_device: int = -1) -> JsonDict:
        parameter_filename = 'allennlp/seq2seq.json'
        serialization_dir = 'retrained'
        subprocess.check_call(['mkdir', '-p', serialization_dir])
        params = Params.from_file(parameter_filename)

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(self._model.vocab)

        parameters = [[n, p] for n, p in self._model.named_parameters() if p.requires_grad]
        trainer_params = params.pop('trainer')
        optimizer = Optimizer.from_params(parameters, trainer_params.pop("optimizer"))

        # all_datasets = datasets_from_params(params)
        # train_data = all_datasets['train']
        trainer = SimpleTrainer(self._model, optimizer, [], iterator)
        Interpreter(self._model, self._dataset_reader, trainer).cmdloop()

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict], cuda_device: int = -1) -> List[JsonDict]:
        raise Exception('Unimplemented')


@Predictor.register('simple_seq2seq')
class SimpleSeq2SeqPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source), {}
