from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor
from typing import List, Tuple
import json
from allennlp.data.dataset import Batch
from allennlp.common import Registrable
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq

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
            instance_output = self._model.beam_search(instance.fields['source_tokens'].as_array(), bestk=100)
            output_string += instance_output
        return output_string

