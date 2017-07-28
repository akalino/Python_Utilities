import collections
import random
import re
import string

import numpy as np
import pandas as pd

# Initialize the data_index to zero
data_index = 0


def create_contexts(data, batch_size, num_skips, skip_window):
    # Make the data index a global variable to be used in each batching step
    global data_index
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span -1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack for the next batching
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


def to_unicode(_term):
    """ A function to try many ways to unicode a string.
    :param _term: An individual term.
    :return: Decoded input string.
    """
    if isinstance(_term, unicode):
        return _term
    elif isinstance(_term, str):
        try:
            if _term.startswith('\xff\xfe'):
                return _term.decode('utf-16-le')
            elif _term.startswith('\xfe\xff'):
                return _term.decode('utf-16-be')
            else:
                return _term.decode('utf-8')
        except UnicodeDecodeError:
            return _term.decode('latin-1')
    else:
        try:
            return unicode(_term)
        except UnicodeError:
            return to_unicode(str(_term))
        except TypeError:
            if hasattr(_term, '__unicode__'):
                return _term.__unicode__()


def remove_pii(_text):
    """ A function to strip the documents of the CFPB scrubbed PII.
    :param _text: An individual document.
    :return: The same document without the masking strings.
    """
    return re.sub(r'\s*X+', '', _text)


def remove_punctuation(_text, _split):
    """ Using the string translate functionality to strip bad characters.
    :param _text: An individual document.
    :param _split: Binary flag to split string to tokens.
    :return: A cleaned document.
    """
    _text = _text.translate(None, string.punctuation)
    if _split:
        _text = _text.strip('\n')
        _text = _text.split(' ')
    return _text


def clean_text(_current_vocab, _string):
    cleaned = filter(bool, remove_punctuation(remove_pii(_string), True))
    for w in cleaned:
        _current_vocab.append(to_unicode(w.lower()))
    return _current_vocab


def create_vocabulary(_path):
    vocabulary = []
    cfpb_raw = pd.read_csv(_path)
    column_list = cfpb_raw.columns
    new_columns = [x.lower().replace(' ', '_') for x in column_list]
    cfpb_raw.columns = new_columns
    texts = cfpb_raw[cfpb_raw['consumer_complaint_narrative'].notnull()]
    texts.reset_index(inplace=True, drop=True)
    for k in range(len(texts)):
        vocabulary = clean_text(vocabulary, texts['consumer_complaint_narrative'][k])
    return vocabulary


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
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
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
