#!/usr/bin/env python3

from sys import stderr
import torch

# available models at https://huggingface.co/pytorch-transformers/pretrained_models.html
# if you want to use ELMo then use modelname elmo or elmo-incremental

class ContextualizedEmbedder:


  def __init__(self, model_name):
    self.model_name      = model_name
    self._tokenizer      = None
    self._model          = None
    self._special_tokens = None
    self._pad_token      = None

    res = self.embed_batch(["This is some sentence"])
    self.dim = res[0][0][0].size
    self.layers = len(res[0])

  # lazy building of tagger
  @property
  def tokenizer(self):
    if self._tokenizer is None:
      from pytorch_transformers import AutoTokenizer
      self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    return self._tokenizer

  # lazy building of model
  @property
  def model(self):
    if self._model is None:
      from pytorch_transformers import AutoModel
      self._model = AutoModel.from_pretrained(self.model_name, output_hidden_states=True)
      self._model.eval()
    return self._model

  @property
  def pad_token(self):
    if self._pad_token is None:
      if "bert" in self.model_name:
        self._pad_token = self.tokenizer.pad_token
      else:
        self._pad_token = "[PAD]"
    return self._pad_token

  @property
  def special_tokens(self):
    if self._special_tokens is None:
      if "bert" in self.model_name:
        self._special_tokens = [self.tokenizer.cls_token, self.tokenizer.sep_token]
      else:
        self._special_tokens = [self.tokenizer.bos_token, self.tokenizer.eos_token]
    return self._special_tokens

  # helpers
  @staticmethod
  def pad(llist, pad_token, to_length):
    return llist + [pad_token] * (to_length-len(llist))

  def extract(self, tensor, token_ids, sent):
    tokenizer = self.tokenizer
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    curr_pos = 0
    output = []
    for word in sent.split(" "):
      while tokens[curr_pos] in self.special_tokens:
        curr_pos += 1
      output.append(tensor[curr_pos])
      to_skip = len(tokenizer.encode(word, add_special_tokens=False))
      curr_pos += to_skip
    return output

  def tokenize(self, sent):
    if "bert" in self.model_name:
      return self.tokenizer.encode(sent, add_special_tokens=True)
    else:
      return [token for word in sent.split(" ") for token in self.tokenizer.encode(word)]

  # embedding_batches
  # returns output[sent_id][layer_id][word_id]
  def embed_batch(self, sents):
    pad       = ContextualizedEmbedder.pad
    sents     = [ContextualizedEmbedder.penn_to_normal(x) for x in sents]
    tokenized = [self.tokenize(x) for x in sents]
    max_len   = max([len(x) for x in tokenized])
    pad_id    = self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]
    padded    = [pad(x, pad_id, max_len) for x in tokenized]
    with torch.no_grad():
      # layers[-1] is the top layer
      # layers[ 0] is the bottom layer
      if "bert" in self.model_name:
        mask   = [pad([1 for _ in x], 0, max_len) for x in tokenized]
        layers = self.model(torch.tensor(padded), attention_mask=torch.tensor(mask))[2]
      else:
        layers = self.model(torch.tensor(padded))[2]
    return [
             [
               self.extract(layer.numpy()[i], tokenized[i], sent)
               for layer in layers
             ]
             for i, sent in enumerate(sents)
           ]

  @staticmethod
  def penn_to_normal(sent):
    words = sent.split(" ")
    for i, word in enumerate(words):
      if word == "-LRB-":
        words[i] = "("
      elif word == "-RRB-":
        words[i] = ")"
      elif word == "\\/":
        words[i] = "/"
      elif word == "\\*":
        words[i] = "*"
      # elif i>1 and word == "n't" and words[i-1] not in ["can", "won"]:
      #   words[i] = "'t"
      #   words[i-1] += "n"
    return " ".join(words)

class ELMoEmbedderAccessor:

  import numpy as np

  def __init__(self, model_name):
    self.initialized = False
    self.layers = 3
    if model_name == "elmo":
      self.is_incremental = False
      self.dim = 2*512
    elif model_name == "elmo-incremental":
      self.is_incremental = True
      self.dim = 512
    else:
      raise Exception("unknown elmo type %s"%model_name)

  def initializeELMo(self):
    if not self.initialized:
      print("loading ELMo START", file=stderr)
      from allennlp.commands.elmo import ElmoEmbedder
      options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
      weight_file  = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
      self.embedder = ElmoEmbedder(options_file=options_file, weight_file=weight_file)
      self.embedder.elmo_bilm.eval()
      self.initialized = True
      print("loading ELMo DONE", file=stderr)

  # embedding_batches
  # returns output[sent_id][layer_id][word_id]
  def embed_batch(self, batch):
    self.initializeELMo()
    ress = self.embedder.embed_sentences(batch)
    all_vectors = []


    return [
             [
               [
                 res[layer, word_position, :512] if self.is_incremental else res[layer, word_position, :]
                 for word_position in range(len(sent))
               ]
               for layer in [0, 1, 2]
             ]
             for sent, res in zip(batch, ress)
           ]

def constructEmbedder(model_name):
  if model_name.startswith("elmo"):
    return ELMoEmbedderAccessor(model_name)
  else:
    return ContextualizedEmbedder(model_name)

# y = constructEmbedder("bert-base-uncased")
# y = constructEmbedder("bert-large-cased-whole-word-masking")
# with open("tmp/ccg_extracted/train.words") as fh:
#   for line in fh:
#     sent = line.rstrip()
#     print(sent)
#     res = y.embed_batch([sent])

# y = constructEmbedder("elmo")
# y = constructEmbedder("bert-base-uncased")
# y = constructEmbedder("distilbert-base-uncased")
# y = constructEmbedder("roberta-base")
# y = constructEmbedder("bert-base-multilingual-cased")

# y = constructEmbedder("gpt2")
# res = y.embed_batch(["This is some sentence car", "this too"])
# print(res)
# print("dimension %d"%y.dim)
# print("layers %d"%y.layers)

