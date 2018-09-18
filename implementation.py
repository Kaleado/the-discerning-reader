import tensorflow as tf
import re

from pathlib import Path
import pickle as pk
import glob

BATCH_SIZE = 16
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
NUM_LAYERS = 1

# TODO: find useful set of stop words online
# stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
#                   'there', 'about', 'once', 'during', 'out', 'very', 'having',
#                   'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
#                   'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
#                   'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
#                   'each', 'the', 'themselves', 'below', 'are', 'we',
#                   'these', 'your', 'his', 'through', 'don', 'me', 'were',
#                   'her', 'more', 'himself', 'this', 'down', 'should', 'our',
#                   'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
#                   'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
#                   'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
#                   'yourselves', 'then', 'that', 'because', 'what', 'over',
#                   'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
#                   'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
#                   'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
#                   'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
#                   'how', 'further', 'was', 'here', 'than'}) # TODO: remove some words from here

stop_words = set({'a', 'about', 'above', 'across', 'after', 'again', 'against', 
'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 
'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 
'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 
'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 
'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 
'becomes', 'been', 'before', 'began', 'behind', 'being', 'beings', 
'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 
'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 
'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently',
'do', 'does', 'done', 'down', 'downed', 'downing', 'downs', 'during', 'e',
'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough',
'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything',
'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 
'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully',
'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general',
'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going',
'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group',
'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 
'having', 'he', 'her', 'here', 'herself', 'high', 'higher',
'highest', 'him', 'himself', 'his', 'how', 'however', 'i',
'if', 'important', 'in', 'interest', 'interested', 'interesting',
'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 
'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows',
'l', 'large', 'largely', 'last', 'later', 'latest', 'least',
'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 
'longest', 'm', 'made', 'make', 'making', 'man', 'many',
'may', 'me', 'member', 'members', 'men', 'might', 'more',
'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my',
'myself', 'n', 'necessary', 'need', 'needed', 'needing', 
'needs', 'never', 'new', 'newer', 'newest', 'next', 'no',
'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere',
'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older',
'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 
'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 
'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 
'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 
'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 
'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 
'rather', 'really', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 
'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 
'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 
'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 
'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 
'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 
'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 
'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 
'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 
'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 
'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 
'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 
'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 
'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 
'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 
'youngest', 'your', 'yours', 'z'})

    #TODO: make a dictionary of words that cannot be converted to embeddings and discard them
    # rather than just having a 0 vector,

    # lemming and stemming

def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    If loaded for the first time, serialize the final dict for quicker loading.
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """

    emmbed_file = Path("./embeddings.pkl")
    if emmbed_file.is_file():
        # embeddings already serialized, just load them
        print("Local Embeddings pickle found, loading...")
        with open("./embeddings.pkl", 'rb') as f:
            return pk.load(f)
    else:
        # create the embeddings
        print("Building embeddings dictionary...")
        data = open("glove.6B.50d.txt", 'r', encoding="utf-8")
        embeddings = [[0] * EMBEDDING_SIZE]
        word_index_dict = {'UNK': 0}  # first row is for unknown words
        index = 1
        for line in data:
            splitLine = line.split()
            word = tf.compat.as_str(splitLine[0])
            embedding = [float(val) for val in splitLine[1:]]
            embeddings.append(embedding)
            word_index_dict[word] = index
            index += 1
        data.close()

        # pickle them
        with open('./embeddings.pkl', 'wb') as f:
            print("Creating local embeddings pickle for faster loading...")
            # Pickle the 'data' dictionary using the highest protocol available.
            pk.dump((embeddings, word_index_dict), f, pk.HIGHEST_PROTOCOL)

    return embeddings, word_index_dict

glove_array, glove_dict = load_glove_embeddings()

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """

    # Convert to lowercase.
    processed_review = review.lower()

    # Remove HTML tags
    processed_review = re.sub("<br />", " ", processed_review)

    # Replace n't with not, so we can learn not
    # processed_review = re.sub("n't", " not", processed_review)

    # Remove punctuation by replacing punctuation with spaces.
    processed_review = re.sub("[\.,'\"\(\)-=+_!@#$%\^&\*]", " ", processed_review)

    # Remove excess spaces.
    processed_review = re.sub(" +", " ", processed_review)

    # Split at each word boundary.
    processed_review = processed_review.split(" ")

    # Remove stop words and words without embeddings
    processed_review = [it for it in processed_review if it not in stop_words and it in glove_dict]

    return processed_review

def get_lstm_cell(lstm_size,dropout_keep_prob):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=dropout_keep_prob)
    return dropout_cell


def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    # Nick: To confirm: what should the shape of the input and output tensors be?
    learning_rate = 0.001 # Nick: it feels weird putting this logic in here.
    # lstm_size = MAX_WORDS_IN_REVIEW # Nick: Also arbitrary, unsure if more units = better.
    lstm_size = 16 # temp for quick testing
    input_data = tf.placeholder(tf.float32,
                                [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],
                                "input_data")

    labels = tf.placeholder(tf.float32, [None, 2], "labels")
    dropout_keep_prob = tf.placeholder_with_default(input=0.6, shape=[], name="dropout_keep_prob") # Nick: To do.

    # Nick: Here we build the network itself. I assume this will involve using
    # LSTM cells from tf.nn.rnn_cell.LSTMCell.
    # lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
    # final_input_sequence = tf.unstack(input_data, MAX_WORDS_IN_REVIEW, 1)
    # rnn_output, state = tf.nn.static_rnn(lstm_cell, final_input_sequence, dtype=tf.float32)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=dropout_keep_prob)
    cell = dropout_cell


    # cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell(lstm_size, dropout_keep_prob) for _ in range(NUM_LAYERS)])
    # cell = tf.nn.rnn_cell.MultiRNNCell([dropout_cell] * NUM_LAYERS)
    # initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
    #final_input_sequence = tf.unstack(input_data, MAX_WORDS_IN_REVIEW, 1)
    rnn_output, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    # rnn_output, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32, initial_state=initial_state)

    # weights = tf.truncated_normal_initializer(stddev=0.1)
    # biases = tf.zeros_initializer()
    
    # fc_units = lstm_size * 2

    # dense = tf.contrib.layers.fully_connected(rnn_output[:, -1],
    #             num_outputs = lstm_size,
    #             # num_outputs = fc_units,
    #             activation_fn = tf.sigmoid,
    #             weights_initializer = weights,
    #             biases_initializer = biases)
    
    # dense = tf.contrib.layers.dropout(dense, dropout_keep_prob)

    # Nick: We need some way of converting the rnn_output to be of shape
    # matching the labels.
    weight = tf.get_variable("weight", [lstm_size, 2], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer, trainable=True)
    # last = rnn_output[-1]
    # last = dense
    # last = dense[-1,lstm_size]
    last = rnn_output[:, -1, :]
    # last = dense[:, -1, :]
    # rnn_output = tf.transpose(rnn_output, [1, 0, 2])
    # last = tf.gather(rnn_output, int(rnn_output.get_shape()[0]) - 1)
    # logits = tf.matmul(last, weight)
    # last = tf.reshape(rnn_output, [int(rnn_output.get_shape()[0]), int(input_data.get_shape()[1])])
    # logits = tf.matmul(last, weight)
    logits = tf.matmul(tf.reshape(last, [-1, lstm_size]), weight)
    preds = tf.nn.softmax(logits)

    # Nick: Should we be using softmax?
    batch_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    batch_loss = tf.reduce_mean(batch_cross_entropy, name="loss") # | || || |_

    # Nick: we calculate the accuracy, this is just copied from asst1 so I have
    # no idea if this is right.
    correct_preds_op = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32), name="accuracy")

    # Nick: This is also arbitrarily chosen.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(batch_loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, batch_loss
