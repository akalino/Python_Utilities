import argparse
import pickle

import tensorflow as tf

from data_utilities import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_word_vectors",
                                     description="Creates word2vec model for given list of tokens")
    parser.add_argument('-d', '--embedding_dimension', nargs='?', type=int, required=True,
                        help='The final vector embedding dimension',
                        dest='ed')
    parser.add_argument('-w', '-write', action="store_true", required=False,
                        help='Save the final embeddings and lexicon as pickled objects',
                        dest='w')

    args = parser.parse_args()

    # Define the model inputs
    data_index = 0
    embedding_dim = args.ed
    batch_size = 128
    neg_sampled = 50
    lr = 0.01
    num_skips = 2
    skip_window = 1

    # Create the dataset
    vocabulary = create_vocabulary('~/Documents/DeepClassification/TF/Code/Data/Consumer_Complaints.csv')
    print('Vocabulary total is {s}'.format(s=len(list(set(vocabulary)))))
    vocab_size = len(list(set(vocabulary)))
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocab_size)
    if args.w:
        pickle.dump(reverse_dictionary, open('lexicon.pkl', 'wb'))
    batch, labels = create_contexts(data, batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],
              '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    # Build the Tensorflow graph
    graph = tf.Graph()

    with graph.as_default():
        # Define the embedding matrix
        # Note these are variables because they will be updated
        embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0))

        out_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_dim],
                                                      stddev=1.0 / np.sqrt(embedding_dim)))
        out_bias = tf.Variable(tf.zeros([vocab_size]))

        # Define skip-gram model
        # Assumption - feeding in a batch of integer (hence int32) word representations
        # Note these are placeholders because they haven't been provided an actual value yet
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Get current embedding from lookup table
        embed = tf.nn.embedding_lookup(embedding, train_inputs)

        # Write the loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=out_weights,
                                             biases=out_bias,
                                             labels=train_labels,
                                             inputs=embed,
                                             num_sampled=neg_sampled,
                                             num_classes=vocab_size))

        # And do SGD on the loss
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embeddings = embedding / norm

        init = tf.global_variables_initializer()

    # Start the training
    num_steps = 100001

    with tf.Session(graph=graph) as session:
        init.run()
        print('Initialized')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = create_contexts(data, batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step {s}: {l}'.format(s=step,
                                                             l=average_loss))
                average_loss = 0
        final_embeddings = normalized_embeddings.eval()
        print('Final embedding shape is {e}'.format(e=final_embeddings.shape))
        if args.w:
            final_embeddings.dump('embeddings_{d}.pkl'.format(d=embedding_dim))
