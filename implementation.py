import tensorflow as tf
import re

BATCH_SIZE = 128 # larger batch size tends to work better
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
NUM_LAYERS = 2 # more than 2 doesn't change much
LEARNING_RATE = 0.001
LSTM_SIZE = 32

# Stop words list taken from: https://github.com/ravikiranj/twitter-sentiment-analyzer/blob/master/data/feature_list/stopwords.txt
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

    # Remove punctuation by replacing punctuation with spaces.
    processed_review = re.sub("[\.,'\"\(\)-=+_!@#$%\^&\*]", " ", processed_review)

    # Remove excess spaces.
    processed_review = re.sub(" +", " ", processed_review)

    # Split at each word boundary.
    processed_review = processed_review.split(" ")

    # Remove stop words.
    processed_review = [it for it in processed_review if it not in stop_words]

    return processed_review[-MAX_WORDS_IN_REVIEW:-1]

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

    input_data = tf.placeholder(tf.float32,
                                [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],
                                "input_data")

    labels = tf.placeholder(tf.float32, [None, 2], "labels")
    dropout_keep_prob = tf.placeholder_with_default(0.6, shape=[], name="dropout_keep_prob") # Nick: To do.

    # Nick: Here we build the network itself. I assume this will involve using
    # LSTM cells from tf.nn.rnn_cell.LSTMCell.
    # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_SIZE);
    # dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
    #                                              output_keep_prob=dropout_keep_prob)
    #rnn_output, state = tf.nn.dynamic_rnn(lstm_cell, input_data, dtype=tf.float32)
    ###
    rnn_layers = \
    [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(LSTM_SIZE),
                                   output_keep_prob=dropout_keep_prob) for x in range(NUM_LAYERS)]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    rnn_output, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=input_data, dtype=tf.float32)
    ###


    # Nick: We need some way of converting the rnn_output to be of shape
    # matching the labels.
    weight = tf.get_variable("weight", [LSTM_SIZE, 2], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer, trainable=True)
    #last = rnn_output[-1] # tf.gather(rnn_output, int(rnn_output.get_shape()[0]) - 1)
    last = rnn_output[:, -1, :]

    # Add dense layer here.
    dense_layer = tf.layers.dense(last, 2)

    logits = dense_layer
    preds = tf.nn.softmax(logits)
    #logits = tf.matmul(tf.reshape(last, [-1, LSTM_SIZE]), weight)
    #preds = tf.nn.softmax(logits)

    # Nick: Should we be using softmax?
    batch_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    batch_loss = tf.reduce_mean(batch_cross_entropy, name="loss") # | || || |_

    # Nick: we calculate the accuracy, this is just copied from asst1 so I have
    # no idea if this is right.
    correct_preds_op = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32), name="accuracy")

    # Nick: This is also arbitrarily chosen.
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(batch_loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, batch_loss
