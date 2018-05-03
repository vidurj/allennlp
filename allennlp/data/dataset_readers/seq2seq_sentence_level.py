from typing import Dict
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TrivialTokenIndexer
from allennlp.prepare_seq2seq_data import is_strict_num
from allennlp.data.tokenizers.word_stemmer import PorterStemmer
import random
from allennlp.prepare_seq2seq_data import standardize_question, standardize_logical_form_with_validation
import sys


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("seq2seq_sentence_level")
class Seq2SeqSentenceLevelDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._stemmer = PorterStemmer()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (line, line_num + 1))
                source_sequence, target_sequence = line_parts
                try:
                    yield self.text_to_instance(source_sequence, target_sequence)
                except AssertionError as e:
                    print(e)
                    print(sys.exc_info()[0])
                    pass
    @overrides
    def text_to_instance(self, raw_source_string: str,
                         _target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        # TODO set randomize to True
        source_string, num_to_token = standardize_question(raw_source_string, copy_mechanism=False, randomize=False)
        sentences = source_string.split('<sentence_end>')
        if _target_string is not None:
            targets = _target_string.split('<sentence_end>')
            assert len(sentences) == len(targets), (sentences, targets)
        else:
            targets = [None for _ in sentences]
        tag_to_field = {}
        var_assignments = {}
        type_assignments = {}
        print('raw source string:', raw_source_string)
        for sentence_number, (sentence, raw_target_string) in enumerate(zip(sentences, targets)):
            print('sentence:', sentence)
            tokenized_source = self._source_tokenizer.tokenize(sentence)
            assert self._source_add_start_token

            if self._source_add_start_token:
                tokenized_source.insert(0, Token(START_SYMBOL))
            tokenized_source.append(Token(END_SYMBOL))
            source_field = TextField(tokenized_source, self._source_token_indexers)
            tag_to_field[str(sentence_number) + '_source_tokens'] = source_field
            if target_string is not None:
                # TODO set randomize to True
                # target_string, _ = standardize_logical_form_with_validation(raw_target_string,
                #                                                             num_to_token,
                #                                                             randomize=False,
                #                                                             var_assignments=var_assignments,
                #                                                             type_assignments=type_assignments)
                print('raw target string:', raw_target_string)
                print('target string:', target_string)
                tokenized_target = self._target_tokenizer.tokenize(target_string)
                tokenized_target.insert(0, Token(START_SYMBOL))
                tokenized_target.append(Token(END_SYMBOL))
                target_field = TextField(tokenized_target, self._target_token_indexers)
                tag_to_field[str(sentence_number) + '_target_tokens'] = target_field
        print('-' * 70)
        return Instance(tag_to_field)

    @classmethod
    def from_params(cls, params: Params) -> 'Seq2SeqSentenceLevelDatasetReader':
        source_tokenizer_type = params.pop('source_tokenizer', None)
        source_tokenizer = None if source_tokenizer_type is None else Tokenizer.from_params(
            source_tokenizer_type)
        target_tokenizer_type = params.pop('target_tokenizer', None)
        target_tokenizer = None if target_tokenizer_type is None else Tokenizer.from_params(
            target_tokenizer_type)
        source_indexers_type = params.pop('source_token_indexers', None)
        source_add_start_token = params.pop_bool('source_add_start_token', True)
        if source_indexers_type is None:
            source_token_indexers = None
        else:
            source_token_indexers = TokenIndexer.dict_from_params(source_indexers_type)
        target_indexers_type = params.pop('target_token_indexers', None)
        if target_indexers_type is None:
            target_token_indexers = None
        else:
            target_token_indexers = TokenIndexer.dict_from_params(target_indexers_type)
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return Seq2SeqSentenceLevelDatasetReader(source_tokenizer=source_tokenizer,
                                    target_tokenizer=target_tokenizer,
                                    source_token_indexers=source_token_indexers,
                                    target_token_indexers=target_token_indexers,
                                    source_add_start_token=source_add_start_token,
                                    lazy=lazy)
