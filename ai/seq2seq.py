import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import os

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

MAX_LENGTH = 10
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

model_name  = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

loadFileName = None
checkpoint_iter = 4000

# Splits each line of the file to create lines and conversations

class EncoderRNN(nn.Module):
  def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
    super(EncoderRNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.embedding = embedding
    # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
    #   because our input size is a word embedding with number of features == hidden_size
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
      embedded = self.embedding(input_seq)
      packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
      outputs, hidden = self.gru(packed, hidden)
      outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
      outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
      return outputs, hidden

class Attn(nn.Module):
  def __init__(self, method, hidden_size):
    super(Attn, self).__init__()
    self.method = method
    if self.method not in ['dot', 'general', 'concat']:
      raise ValueError(self.method, "is not appropriate attention method.")
    self.hidden_size = hidden_size
    if self.method == 'general':
      self.attn = nn.Linear(self.hidden_size, hidden_size)
    elif self.method == 'concat':
      self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
      self.v = nn.Parameter(torch.FloatTensor(hidden_size))
  
  def dot_score(self, hidden, encoder_output):
    return torch.sum(hidden * encoder_output, dim=2)
  
  def general_score(self, hidden, encoder_output):
    energy = self.attn(encoder_output)
    return torch.sum(hidden * energy, dim = 2)
  
  def concat_score(self, hidden, encoder_output):
    energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
    return torch.sum(self.v * energy, dim=2)
  
  def forward(self, hidden, encoder_outputs):
    if self.method == 'general':
      attn_energies = self.general_score(hidden, encoder_outputs)
    elif self.method == 'concat':
      attn_energies = self.concat_score(hidden, encoder_outputs)
    elif self.method == 'dot':
      attn_energies = self.dot_score(hidden, encoder_outputs) 
    attn_energies = attn_energies.t()
    return F.softmax(attn_energies, dim=1).unsqueeze(1)
  
class LuongAttnDecoderRNN(nn.Module):
  def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
    super(LuongAttnDecoderRNN, self).__init__()

    self.attn_model = attn_model
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout = dropout

    self.embedding = embedding
    self.embedding_dropout = nn.Dropout(dropout)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
    self.concat = nn.Linear(hidden_size * 2, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

    self.attn = Attn(attn_model, hidden_size)

  def forward(self, input_step, last_hidden, encoder_ouputs):
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    rnn_output, hidden = self.gru(embedded, last_hidden)
    attn_weights = self.attn(rnn_output, encoder_ouputs)
    context = attn_weights.bmm(encoder_ouputs.transpose(0,1))
    rnn_output = rnn_output.squeeze(0)
    context = context.squeeze(1)
    concat_input = torch.cat((rnn_output, context), 1)
    concat_output = torch.tanh(self.concat(concat_input))
    output = self.out(concat_output)
    output = F.softmax(output, dim=1)
    return output, hidden


def maskNLLLoss(inp, target, mask):
  nTotal = mask.sum()
  crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1,1)).squeeze(1))
  loss = crossEntropy.masked_select(mask).mean()
  loss = loss.to(device)
  return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  input_variable = input_variable.to(device)
  target_variable = target_variable.to(device)
  mask = mask.to(device)

  lengths = lengths.to("cpu")

  loss = 0
  print_losses = []
  n_totals = 0

  encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
  decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
  decoder_input = decoder_input.to(device)
  decoder_hidden = encoder_hidden[:decoder_hidden.n_layers]

  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  if use_teacher_forcing:
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder(
        decoder_input, decoder_hidden, encoder_outputs
      )
      decoder_input = target_variable[t].view(1, -1)
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
      loss += mask_loss
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal
  else:
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder(
        decoder_input, decoder_hidden, encoder_outputs
      )

      _, topi = decoder_output.topk(1)
      decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
      decoder_input = decoder_input.to(device)

      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
      loss += mask_loss
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal
  
  loss.backward()

  _ = nn.utils.clip_grad_norm(encoder.parameters(), clip)
  _ = nn.utils.clip_grad_norm(decoder.parameters(), clip)

  encoder_optimizer.step()
  decoder_optimizer.step()

  return sum(print_losses) / n_totals

def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, 
embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every,
save_every, clip, corpus_name, loadFilename):
  training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]

  # Initialization
  print('Initialization...')
  start_iteration = 1
  print_loss= 0
  if loadFilename:
    start_iteration =  checkpoint['iteration'] + 1
  
  # Training loop
  print('Training...')
  for iteration in range(start_iteration, n_iteration + 1):
    training_batches = training_batches[iteration - 1]
    input_variable, lengths, target_variable, mask, max_target_len = training_batches
    loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
    print_loss += loss
    if (iteration % print_every == 0):
      print_loss_avg = print_loss / print_every
      print("Iteration: {}; Percent complete: {:.1f}; Average loss:{:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
      print_loss = 0

    if (iteration % save_every == 0):
      directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
      if not os.path.exist(directory):
        os.makedirs(directory)
      torch.save({
        'iteration': iteration,
        'en': encoder.save_dict(),
        'de': decoder.save_dict(),
        'en_opt': encoder_optimizer.state_dict(),
        'de_opt': decoder_optimizer.state_dict(),
        'loss': loss,
        'voc_dict': voc.__dict__,
        'embedding': embedding.state_dict()
      }, os.path.join(directory, '{}_{}.tar'. format(iteration, 'checkpoint')))

class GreedySearchDecoder(nn.Module):
  def __init__(self, encoder, decoder):
    super(GreedySearchDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, input_seq, input_length, max_length):
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    all_scores = torch.zeros([0], device=device)
    for _ in range(max_length):
      decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
      decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
      all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
      all_scores = torch.cat((all_scores, decoder_scores), dim=0)
      decoder_input = torch.unzqueeze(decoder_input, 0)
    return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
  indexes_batch = [indexesFromSentence(voc, sentence)]
  lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
  input_batch = torch.LongTensor(indexes_batch).transpose(0, -1)
  input_batch = input_batch.to(device)
  lengths = lengths.to("cpu")
  tokens, scores = searcher(input_batch, lengths, max_length)
  decoded_words = [voc.index2word[token.item()] for token in tokens]
  return decoded_words

def evalutateInput(encoder, decoder, searcher, voc):
  input_sentence = ''
  while(1):
    try:
      input_sentence = input('> ')
      if input_sentence == 'q' or input_sentence == 'quit': break
      input_sentence = normalizeString(input_sentence)
      output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
      output_words[:] = [x for x in output_words if not (x == 'EOS' or  x == 'PAD')]
      print('Bot:', ' '.join(output_words))
    except KeyError:
      print("Error: Encountered unknown word.")


# Configure models

if loadFileName:
  # If loading on same machine the model was trained on
  checkpoint = torch.load(loadFileName)
  # If loading a model trained on GPU to CPU
  #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
  encoder_sd = checkpoint['en']
  decoder_sd = checkpoint['de']
  encoder_optimizer_sd = checkpoint['en_opt']
  decoder_optimizer_sd = checkpoint['de_opt']
  embedding_sd = checkpoint['embedding']
  voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFileName:
  embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFileName:
  encoder.load_state_dict(encoder_sd)
  decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFileName:
  encoder_optimizer.load_state_dict(encoder_optimizer_sd)
  decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
  for k, v in state.items():
    if isinstance(v, torch.Tensor):
      state[k] = v.cuda()

for state in decoder_optimizer.state.values():
  for k, v in state.items():
    if isinstance(v, torch.Tensor):
      state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
  embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
  print_every, save_every, clip, corpus_name, loadFilename)


# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
# evaluateInput(encoder, decoder, searcher, voc)