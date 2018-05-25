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
import traceback
from allennlp.prepare_seq2seq_data import standardize_logical_form_with_validation, \
    standardize_question
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL

@Predictor.register('simple_seq2seq')
class SimpleSeq2SeqPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source" : source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source), {}


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
