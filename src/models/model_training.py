import pandas as pd
import tensorflow as tf
from preprocess_data import TextCleanTransformer


# Setting parameters
vocab_size = 20000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'
training_fraction = 0.8
num_epochs = 30


def train_model(data):
    sentences = data['text_clean'].values
    labels = data['target'].values

    training_size = int(training_fraction * data.shape[0])

    training_sentences = sentences[:training_size]
    training_labels = labels[:training_size]

    test_sentencs = sentences[training_size:]
    test_labels = labels[training_size:]

    vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size,
                                                        standardize='lower_and_strip_punctuation',
                                                        split='whitespace',
                                                        ngrams=None,
                                                        output_mode='int',
                                                        output_sequence_length=max_length,
                                                        pad_to_max_tokens=False
                                                        )
    vectorize_layer.adapt(training_sentences, batch_size=32)

    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    model.fit(training_sentences, training_labels,
              epochs=num_epochs, validation_data=(test_sentencs, test_labels),
              )
    model.save('models/my_model')


cleaner = TextCleanTransformer()
data_train = pd.read_csv('data/train.csv')
data_train['text_clean'] = data_train['text'].apply(cleaner.preprocess_sentence)
train_model(data_train)
