from __future__ import print_function
import collections, math, os, random, zipfile, string, pickle
import numpy as np
import tensorflow as tf
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

text_corpus = 'drew'

if text_corpus=='text_8':
    url = 'http://mattmahoney.net/dc/'

    def maybe_download(filename, expected_bytes):
      """Download a file if not present, and make sure it's the right size."""
      if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
      statinfo = os.stat(filename)
      if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
      else:
        print(statinfo.st_size)
        raise Exception(
          'Failed to verify ' + filename + '. Can you get to it with a browser?')
      return filename

    filename = maybe_download('text8.zip', 31344016)
        
    def read_data(filename):
      with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
      return data

    words = read_data(filename)

elif text_corpus=='drew':

    def file_len(fname):
        with open(fname) as f:
            for i, _ in enumerate(f):
                pass
            return i + 1

    def read_data(fname, num_lines, nb_lines):
      data = []
      if num_lines>nb_lines:
          num_lines = nb_lines
          print('too much lines')
      with open(fname, 'r') as f:
        for _ in range(num_lines):
            data.extend(f.readline().translate(string.maketrans("",""), string.punctuation)[:-1].split(" ")) 
      return data 

    path = "../../data/"
    name = "master_clean.txt"
    nb_lines = file_len(os.path.join(path, name))
    words = read_data(os.path.join(path, name), int(1e7), nb_lines)
    
elif text_corpus=='phil':
    fname = "english/master_clean.txt"
    def file_len(fname):
        with open(fname) as f:
            for i, _ in enumerate(f):
                pass
            return i + 1

    def read_data(fname, num_lines, nb_lines):
      data = []
      if num_lines>nb_lines:
          num_lines = nb_lines
          print('too much lines')
      with open(fname, 'r') as f:
        for _ in range(num_lines):
            data.extend(f.readline().translate(string.maketrans("",""), string.punctuation)[:-1].split(" ")) 
      return data 

    path = "english/"
    name = "master_clean.txt"
    nb_lines = file_len(os.path.join(path, name))
    words = read_data(os.path.join(path, name), int(1e7), nb_lines)

print('Data size %d' % len(words))

# English dictionary 171,476 words + 47,156 obsoletes
#vocabulary_size = 100000
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.

data_index = 0

def generate_batch(batch_size, skip_window):
  global data_index
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  batch = np.ndarray(shape=(batch_size,span-1), dtype=np.int32)#batch is now words surrounding each target word
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size):
    target = skip_window  # target label at the center of the buffer
    for j in range(span):
      #adds words before target to batch
      if j < skip_window:
        batch[i, j] = buffer[j]
      #adds words after target to batch
      elif j > skip_window:
        batch[i, j-1] = buffer[j]
    labels[i, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for skip_window in [2, 1]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, skip_window=skip_window)
    print('\nwith skip_window = %d:' % (skip_window))
    print('    batch:', [map(lambda x:reverse_dictionary[x],bi) for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
# NOTE: Maybe consider caption average caption length?
skip_window = 4 # How many words to consider left and right.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 13 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(): #, tf.device('/cpu:0'):

  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size,2*skip_window])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Model.
  # Look up embeddings for inputs.
  # new in CBOW: find mean of all surround words' embeddings
  embed =  tf.reduce_mean(tf.nn.embedding_lookup(embeddings, train_dataset),1)
  target_words=['above','behind','below','between','beyond','inside', \
                'near','outside','over','under','upon','with','within'] 
  target_words=map(lambda x:dictionary[x],target_words)
  target_matrix=[]
  for i in range(batch_size):
    target_matrix.append(target_words)
  # Compute the softmax loss, using a sample of the negative labels each time.
  target_samples = tf.nn.uniform_candidate_sampler(target_matrix,13,13,True,13)
  loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                               train_labels, num_sampled, vocabulary_size, sampled_values=target_samples))

  # Optimizer.
  # Note: The optimizer will optimize the softmax_weights AND the embeddings.
  # This is because the embeddings are defined as a variable quantity and the
  # optimizer's `minimize` method will by default modify all variable quantities 
  # that contribute to the tensor it is passed.
  # See docs on `tf.train.Optimizer.minimize()` for more details.
  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


num_steps = 200001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 20000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()

# Save embeddings and dictionaries
def save_dic(dic, path, name):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

save_dic(dictionary, path, 'dictionary_one_layer_sampling')
save_dic(reverse_dictionary, path, 'reverse_dictionary_one_layer_sampling')
np.save(os.path.join(path, 'embeddings_one_layer_sampling.npy'), final_embeddings)
