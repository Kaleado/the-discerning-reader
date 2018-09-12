import tensorflow as tf
import re

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

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

    return processed_review



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
    num_units = 7 # Nick: Also arbitrary, unsure if more units = better.
    input_data = tf.placeholder(tf.float32,
                                [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],
                                "input_data")

    labels = tf.placeholder(tf.float32, [None, 2], "labels")
    dropout_keep_prob = tf.placeholder(tf.float32, shape=[]) # Nick: To do.

    # Nick: Here we build the network itself. I assume this will involve using
    # LSTM cells from tf.nn.rnn_cell.LSTMCell.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units);
    dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                 input_keep_prob=dropout_keep_prob,
                                                 output_keep_prob=dropout_keep_prob,
                                                 state_keep_prob=dropout_keep_prob)
    #final_input_sequence = tf.unstack(input_data, MAX_WORDS_IN_REVIEW, 1)
    rnn_output, state = tf.nn.dynamic_rnn(lstm_cell, input_data, dtype=tf.float32)

    # Nick: We need some way of converting the rnn_output to be of shape
    # matching the labels.
    weight = tf.get_variable("weight", [num_units, 2], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer, trainable=True)
    #last = rnn_output[-1] # tf.gather(rnn_output, int(rnn_output.get_shape()[0]) - 1)
    last = rnn_output[:, -1, :]
    #print(last.get_shape())
    logits = tf.matmul(tf.reshape(last, [-1, num_units]), weight)
    preds = tf.nn.softmax(logits)

    # Nick: Should we be using softmax?
    batch_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    batch_loss = tf.reduce_mean(batch_cross_entropy, name="loss") # | || || |_

    # Nick: we calculate the accuracy, this is just copied from asst1 so I have
    # no idea if this is right.
    correct_preds_op = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32), name="accuracy")

    # Nick: This is also arbitrarily chosen.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(batch_loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, batch_loss
