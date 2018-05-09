from typing import Dict

import numpy
import numpy as np
import torch
import torch.nn.functional as F
from overrides import overrides
from torch.autograd import Variable
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common import Params
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum
from allennlp.type_checking import valid_next_characters, update_state
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from collections import defaultdict
from copy import deepcopy

"""
And (bool, ...) : bool
Equals (num, num) : bool
Join   (set, set) : set
Value  (set, type) : num
Rate   (set, type, type) : num
Minus  (num, num) : num
Plus   (num, num) : num


"""


def batched_index_select(input_tensor, index_tensor):
    dummy = index_tensor.unsqueeze(2)
    out = input_tensor.gather(2, dummy)  # b x e x f
    return out.squeeze(2)


@Model.register("simple_seq2seq")
class SimpleSeq2Seq(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    This ``SimpleSeq2Seq`` model takes an encoder (:class:`Seq2SeqEncoder`) as an input, and
    implements the functionality of the decoder.  In this implementation, the decoder uses the
    encoder's outputs in two ways. The hidden state of the decoder is initialized with the output
    from the final time-step of the encoder, and when using attention, a weighted average of the
    outputs from the encoder is concatenated to the inputs of the decoder at every timestep.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : int, required
        Length of decoded sequences
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention_function: ``SimilarityFunction``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    scheduled_sampling_ratio: float, optional (default = 0.0)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 attention_function: SimilarityFunction = None,
                 scheduled_sampling_ratio: float = 0.0) -> None:
        super(SimpleSeq2Seq, self).__init__(vocab)
        self._source_embedder = source_embedder
        self._encoder_hidden_dim = 256
        self._encoder_num_layers = 2
        self._encoder = torch.nn.LSTM(source_embedder.get_output_dim(),
                                      self._encoder_hidden_dim,
                                      self._encoder_num_layers,
                                      dropout=0.2,
                                      batch_first=True,
                                      bidirectional=True)  # encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._attention_function = attention_function
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        assert self._start_index != self._end_index
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        self.num_classes = num_classes
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder. Also, if
        # we're using attention with ``DotProductSimilarity``, this is needed.
        self._decoder_output_dim = 2 * self._encoder_hidden_dim
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)
        if self._attention_function:
            self._decoder_attention = Attention(self._attention_function)
            # The output of attention, a weighted average over encoder outputs, will be
            # concatenated to the input vector of the decoder at each time step.
            self._decoder_input_dim = 2 * self._encoder_hidden_dim + target_embedding_dim
        else:
            self._decoder_input_dim = target_embedding_dim
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

    def beam_search(self,  # type: ignore
                    source_tokens,
                    bestk,
                    generate_stray_constraints) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
        """

        sentence_number_to_text_field = defaultdict(dict)
        for key, value in source_tokens.items():
            tokens = key.split('_')
            index = int(tokens[0])
            remaining_key = '_'.join(tokens[1:])
            sentence_number_to_text_field[index][remaining_key] = value

        batch_size = 1
        decoder_zeros = Variable(
            torch.cuda.FloatTensor(batch_size, self._decoder_output_dim).fill_(0))

        sentence_embeddings = []
        source_masks = []
        sentence_to_valid_numbers = []
        for sentence_number in range(len(sentence_number_to_text_field)):
            relevant_text_fields = sentence_number_to_text_field[sentence_number]
            source_tokens = relevant_text_fields['source_tokens']
            source_mask = get_text_field_mask(source_tokens)
            source_masks.append(source_mask)
            embedded_input = self._source_embedder(source_tokens)
            sentence_embeddings.append(embedded_input)
            source_indices = source_tokens['tokens'].data.cpu().numpy()
            valid_numbers = {self.vocab.get_token_from_index(index, 'source_tokens') for index
                             in source_indices[0]}
            valid_numbers = {x for x in valid_numbers if x.startswith('num')}
            valid_numbers.add('(')
            valid_numbers.add('?')
            sentence_to_valid_numbers.append(valid_numbers)

        def encode_sentence(start_decoder_hidden, final_decoder_hidden, start_decoder_context,
                            final_decoder_context, sentence_number):
            start_encoder_hidden = torch.cat(
                [start_decoder_hidden.view(2, 1, self._encoder_hidden_dim),
                 final_decoder_hidden.view((2, 1, self._encoder_hidden_dim))], dim=0)
            start_encoder_context = torch.cat(
                [start_decoder_context.view(2, 1, self._encoder_hidden_dim),
                 final_decoder_context.view((2, 1, self._encoder_hidden_dim))], dim=0)

            embedded_input = sentence_embeddings[sentence_number]
            encoder_outputs, (final_encoder_hidden, final_encoder_context) = \
                self._encoder(embedded_input, (start_encoder_hidden, start_encoder_context))

            start_decoder_hidden = final_encoder_hidden[2:, :, :].view(1, self._decoder_output_dim)
            start_decoder_context = final_encoder_context[2:, :, :].view(1,
                                                                         self._decoder_output_dim)
            return (encoder_outputs, start_decoder_hidden, start_decoder_context)

        (start_encoder_outputs, start_decoder_hidden, start_decoder_context) = encode_sentence(
            decoder_zeros,
            decoder_zeros,
            decoder_zeros,
            decoder_zeros,
            sentence_number=0)
        model = {
            'start_decoder_hidden': start_decoder_hidden,
            'start_decoder_context': start_decoder_context,
            'decoder_hidden': start_decoder_hidden,
            'decoder_context': start_decoder_context,
            'encoder_outputs': start_encoder_outputs,
            'sentence_number': 0,
            'cur_log_probability': 0,
            'action_list': [[START_SYMBOL]],
            'arg_numbers': [0],
            'function_calls': []
        }

        valid_variables = {'var' + str(i) for i in range(20)}
        valid_variables.add('(')
        valid_variables.add('?')
        valid_units = {'unit' + str(i) for i in range(20)}
        models = [model]
        if generate_stray_constraints:
            stopping_point = len(sentence_number_to_text_field) - 1
        else:
            stopping_point = len(sentence_number_to_text_field) - 2
        for cur_length in range(self._max_decoding_steps + 2):
            new_models = []
            for model in models:
                if model['action_list'][-1][-1] == END_SYMBOL:
                    if model['sentence_number'] < stopping_point:
                        model['sentence_number'] += 1
                        (new_encoder_outputs, start_decoder_hidden, start_decoder_context) = \
                            encode_sentence(model['start_decoder_hidden'],
                                            model['decoder_hidden'],
                                            model['start_decoder_context'],
                                            model['decoder_context'],
                                            sentence_number=model['sentence_number'])
                        model['decoder_hidden'] = model['start_decoder_hidden'] = start_decoder_hidden
                        model['decoder_context'] = model['start_decoder_context'] = start_decoder_context
                        model['encoder_outputs'] = new_encoder_outputs
                        model['arg_numbers'] = [0]
                        model['function_calls'] = []
                        model['action_list'].append([START_SYMBOL])
                    else:
                        assert model['sentence_number'] == stopping_point
                        new_models.append(model)
                        continue

                assert len(model['action_list']) == model['sentence_number'] + 1, (len(model['action_list']), model['sentence_number'] + 1)
                assert model['sentence_number'] < len(sentence_number_to_text_field)
                decoder_hidden = model['decoder_hidden']
                decoder_context = model['decoder_context']
                cur_log_probability = model['cur_log_probability']
                source_mask = source_masks[model['sentence_number']]
                valid_numbers = sentence_to_valid_numbers[model['sentence_number']]
                last_prediction_index = self.vocab.get_token_index(model['action_list'][-1][-1], self._target_namespace)
                decoder_input = self._prepare_decode_step_input(
                    Variable(torch.cuda.LongTensor(1).fill_(last_prediction_index)),
                    decoder_hidden,
                    model['encoder_outputs'],
                    source_mask)

                decoder_hidden, decoder_context = self._decoder_cell(decoder_input,
                                                                     (decoder_hidden,
                                                                      decoder_context))

                output_projections = self._output_projection_layer(decoder_hidden)
                class_log_probabilities = \
                    F.log_softmax(output_projections, dim=-1).data.cpu().numpy()[0]
                assert self.vocab.get_vocab_size(self._target_namespace) == len(
                    class_log_probabilities), (self.vocab.get_vocab_size(self._target_namespace),
                                               class_log_probabilities.shape[0])
                valid_actions = valid_next_characters(model['function_calls'],
                                                      model['arg_numbers'],
                                                      model['action_list'][-1][-1],
                                                      valid_numbers,
                                                      valid_variables,
                                                      valid_units)
                seen_new_var = False
                seen_new_unit = False
                seen_actions = set([x for sequence in model['action_list'] for x in sequence])
                for action_index, action_log_probability in enumerate(class_log_probabilities):
                    penalty = 0
                    action = self.vocab.get_token_from_index(action_index, self._target_namespace)
                    if action not in valid_actions:
                        continue

                    if action.startswith('var') and action not in seen_actions:
                        if seen_new_var:
                            continue
                        else:
                            seen_new_var = True

                    if action.startswith('unit') and action not in seen_actions:
                        if seen_new_unit:
                            continue
                        else:
                            seen_new_unit = True

                    if action.startswith('num') and action in seen_actions:
                        penalty += 10

                    if model['action_list'][-1][-1] == '?' and action in seen_actions:
                        continue

                    function_calls, arg_numbers = update_state(model['function_calls'],
                                                               model['arg_numbers'], action)
                    action_list = deepcopy(model['action_list'])
                    action_list[-1].append(action)
                    new_model = {
                        'action_list': action_list,
                        'start_decoder_hidden': model['start_decoder_hidden'],
                        'start_decoder_context': model['start_decoder_context'],
                        'decoder_hidden': decoder_hidden,
                        'decoder_context': decoder_context,
                        'cur_log_probability': action_log_probability + cur_log_probability - penalty,
                        'function_calls': function_calls,
                        'arg_numbers': arg_numbers,
                        'sentence_number': model['sentence_number'],
                        'encoder_outputs': model['encoder_outputs']
                    }
                    new_models.append(new_model)
            assert len(new_models) > 0, (valid_actions, model['action_list'][-1], seen_actions)
            new_models.sort(key=lambda x: - x['cur_log_probability'])
            models = new_models[:bestk]

        models = [model for model in models if model['action_list'][-1] == END_SYMBOL and model['sentence_number'] == stopping_point]
        models.sort(key=lambda x: - x['cur_log_probability'])
        # print('total models', len(models), 'len complete models', len(complete_models))

        # output = '\n'.join([' '.join([token for token in model['action_list'] if token != START_SYMBOL and token != END_SYMBOL]) for model in models])
        # print(' '.join(complete_models[0]['action_list'][1:-1]))
        return [model['action_list'] for model in models]

    def _decode(self, decoder_hidden, decoder_context, max_decoding_steps, encoder_outputs,
                source_mask, targets):
        batch_size = 1
        step_logits = []
        step_probabilities = []
        step_predictions = []
        flag = False
        last_predictions = Variable(source_mask.data.new()
                                    .resize_(batch_size).fill_(self._start_index))
        for timestep in range(max_decoding_steps):
            if targets is not None:
                input_choices = targets[:, timestep]
            elif timestep == max_decoding_steps - 1:
                input_choices = Variable(source_mask.data.new()
                                    .resize_(batch_size).fill_(self._end_index))
            else:
                input_choices = last_predictions
            decoder_input = self._prepare_decode_step_input(input_choices, decoder_hidden,
                                                            encoder_outputs, source_mask)
            decoder_hidden, decoder_context = self._decoder_cell(decoder_input,
                                                                 (decoder_hidden,
                                                                  decoder_context))
            if input_choices.data.cpu()[0] == self._end_index:
                final_decoder_hidden = decoder_hidden
                final_decoder_context = decoder_context
                flag = True
                break
            else:
                # (batch_size, num_classes)
                output_projections = self._output_projection_layer(decoder_hidden)
                # list of (batch_size, 1, num_classes)
                step_logits.append(output_projections.unsqueeze(1))
                class_probabilities = F.softmax(output_projections, dim=-1)
                _, predicted_classes = torch.max(class_probabilities, 1)
                step_probabilities.append(class_probabilities.unsqueeze(1))
                last_predictions = predicted_classes
                # (batch_size, 1)
                step_predictions.append(last_predictions.unsqueeze(1))
        assert flag
        if targets is not None:
            log_probabilities = F.log_softmax(torch.cat(step_logits, 1), dim=2)
            temp = batched_index_select(log_probabilities, targets[:, 1:])
            loss = - torch.sum(temp)
        else:
            loss = None
        return (final_decoder_hidden, final_decoder_context, loss, step_logits, step_probabilities,
                step_predictions)

    @overrides
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        sentence_number_to_text_field = defaultdict(dict)
        for key, value in kwargs.items():
            tokens = key.split('_')
            index = int(tokens[0])
            remaining_key = '_'.join(tokens[1:])
            sentence_number_to_text_field[index][remaining_key] = value

        has_targets = 'target_tokens' in sentence_number_to_text_field[0]

        all_logits = []
        all_probabilities = []
        all_predictions = []
        batch_size = 1
        final_decoder_hidden = Variable(
            torch.cuda.FloatTensor(batch_size, self._decoder_output_dim).fill_(0))
        final_decoder_context = Variable(
            torch.cuda.FloatTensor(batch_size, self._decoder_output_dim).fill_(0))
        start_decoder_hidden = Variable(
            torch.cuda.FloatTensor(batch_size, self._decoder_output_dim).fill_(0))
        start_decoder_context = Variable(
            torch.cuda.FloatTensor(batch_size, self._decoder_output_dim).fill_(0))
        total_loss = Variable(torch.cuda.FloatTensor(1).fill_(0))
        for sentence_number in range(len(sentence_number_to_text_field)):
            relevant_text_fields = sentence_number_to_text_field[sentence_number]
            source_tokens = relevant_text_fields['source_tokens']
            print(' '.join([self.vocab.get_token_from_index(index, 'source_tokens') for index in source_tokens['tokens'].data.cpu().numpy()[0]]))
            source_mask = get_text_field_mask(source_tokens)
            embedded_input = self._source_embedder(source_tokens)
            batch_size, _, _ = embedded_input.size()
            start_encoder_hidden = torch.cat(
                [start_decoder_hidden.view(2, 1, self._encoder_hidden_dim),
                 final_decoder_hidden.view((2, 1, self._encoder_hidden_dim))], dim=0)
            start_encoder_context = torch.cat(
                [start_decoder_context.view(2, 1, self._encoder_hidden_dim),
                 final_decoder_context.view((2, 1, self._encoder_hidden_dim))], dim=0)
            encoder_outputs, (final_encoder_hidden, final_encoder_context) = \
                self._encoder(embedded_input, (start_encoder_hidden, start_encoder_context))
            if has_targets:
                targets = relevant_text_fields['target_tokens']["tokens"]
                print(' '.join([self.vocab.get_token_from_index(index, 'target_tokens') for index in targets.data.cpu().numpy()[0]]))
                max_decoding_steps = targets.size()[1] + self._max_decoding_steps
            else:
                targets = None
                max_decoding_steps = self._max_decoding_steps

            start_decoder_hidden = final_encoder_hidden[2:, :, :].view(1, self._decoder_output_dim)
            start_decoder_context = final_encoder_context[2:, :, :].view(1,
                                                                         self._decoder_output_dim)

            (final_decoder_hidden, final_decoder_context, loss, step_logits, step_probabilities,
             step_predictions) = self._decode(start_decoder_hidden,
                                              start_decoder_context,
                                              max_decoding_steps,
                                              encoder_outputs,
                                              source_mask,
                                              targets)
            all_logits.extend(step_logits)
            all_predictions.extend(step_predictions)
            all_probabilities.extend(step_probabilities)
            if loss is not None:
                total_loss += loss

        logits = torch.cat(all_logits, 1)
        class_probabilities = torch.cat(all_probabilities, 1)
        all_predictions = torch.cat(all_predictions, 1)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities,
                       "predictions": all_predictions, "loss": total_loss}

        return output_dict

    def _prepare_decode_step_input(self,
                                   input_indices: torch.LongTensor,
                                   decoder_hidden_state: torch.LongTensor = None,
                                   encoder_outputs: torch.LongTensor = None,
                                   encoder_outputs_mask: torch.LongTensor = None) -> torch.LongTensor:
        """
        Given the input indices for the current timestep of the decoder, and all the encoder
        outputs, compute the input at the current timestep.  Note: This method is agnostic to
        whether the indices are gold indices or the predictions made by the decoder at the last
        timestep. So, this can be used even if we're doing some kind of scheduled sampling.

        If we're not using attention, the output of this method is just an embedding of the input
        indices.  If we are, the output will be a concatentation of the embedding and an attended
        average of the encoder inputs.

        Parameters
        ----------
        input_indices : torch.LongTensor
            Indices of either the gold inputs to the decoder or the predicted labels from the
            previous timestep.
        decoder_hidden_state : torch.LongTensor, optional (not needed if no attention)
            Output of from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor, optional (not needed if no attention)
            Encoder outputs from all time steps. Needed only if using attention.
        encoder_outputs_mask : torch.LongTensor, optional (not needed if no attention)
            Masks on encoder outputs. Needed only if using attention.
        """
        # input_indices : (batch_size,)  since we are processing these one timestep at a time.
        # (batch_size, target_embedding_dim)
        embedded_input = self._target_embedder(input_indices)
        if self._attention_function:
            # encoder_outputs : (batch_size, input_sequence_length, encoder_output_dim)
            # Ensuring mask is also a FloatTensor. Or else the multiplication within attention will
            # complain.
            encoder_outputs_mask = encoder_outputs_mask.float()
            # (batch_size, input_sequence_length)
            input_weights = self._decoder_attention(decoder_hidden_state, encoder_outputs,
                                                    encoder_outputs_mask)
            # (batch_size, encoder_output_dim)
            attended_input = weighted_sum(encoder_outputs, input_weights)
            # (batch_size, encoder_output_dim + target_embedding_dim)
            return torch.cat((attended_input, embedded_input), -1)
        else:
            return embedded_input

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # relevant_targets = targets.contiguous()  # (batch_size, num_decoding_steps)
        # relevant_mask = target_mask.contiguous()  # (batch_size, num_decoding_steps)
        loss = sequence_cross_entropy_with_logits(logits, targets, target_mask)
        return loss

    @staticmethod
    def _get_accuracy(predictions: torch.LongTensor,
                      targets: torch.LongTensor,
                      target_mask: torch.LongTensor) -> torch.LongTensor:
        targets = targets[:, 1:]
        targets = targets.data.numpy()
        target_mask = target_mask[:, 1:]
        target_mask = target_mask.data.numpy()
        predictions = predictions.data.numpy()
        total = np.sum((targets == predictions) * target_mask)
        return total / np.sum(target_mask)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.data.cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            predicted_tokens = [
                self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'SimpleSeq2Seq':
        source_embedder_params = params.pop("source_embedder")
        source_embedder = TextFieldEmbedder.from_params(vocab, source_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        target_namespace = params.pop("target_namespace", "tokens")
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        target_embedding_dim = params.pop("target_embedding_dim", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        scheduled_sampling_ratio = params.pop_float("scheduled_sampling_ratio", 0.0)
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   source_embedder=source_embedder,
                   encoder=encoder,
                   max_decoding_steps=max_decoding_steps,
                   target_namespace=target_namespace,
                   target_embedding_dim=target_embedding_dim,
                   attention_function=attention_function,
                   scheduled_sampling_ratio=scheduled_sampling_ratio)
