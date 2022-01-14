import pandas as pd
import numpy as np
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.models import load_model
import joblib


imdb = pd.read_csv("IMDB Dataset.csv")
test_data = pd.read_csv("blabla.csv")
nulls = ["n/a", "na", "--", " ", "-", "Not Available","/", "-/" ,".",",", "- -", "/-", "NaN", "nan", "NA", "__", "_", "-_", "(", ")","NaT"]
vocab = 100000
epochs = 2
model_name = 'test.h5'
settings = {'data': imdb,
           'nulls': nulls,
            'vocab': vocab,
            'epochs': epochs,
            'model_name': model_name,

           }


class SpotNan():
    """Generic class that uses a public text data with reviews and enriches it with all kinds of possible null values in any formats someone may encounter
    in his data science projects. The dataset is labeled as null(1) or not null(0) and then an RNN is being trained to address this text classification problem.
    The module can save and load the produced models and the predict method can get a any new dataset as argument to spot the null values.

    Args:
        settings: an initialization dictionary of the form:
        >>> settings = {
        ...     'data': pandas dataframe, df
        ...     'nulls': list, a list of null possible values
        ...     'vocab': int, vocabulary length,
        ...     'epochs': int, epochs for the training part
        ...     'model_name': str, pick a model name
        ... }

    Returns:
        The test dataset of the users choice, after converting all types of string typed nulls to np.nan format.

    Example:
         >>> s = SpotNan(settings)
         >>> s.preprocessing()
         >>> s.train()
         >>> s.save_model()
         >>> s.load_model(model_name)
         >>> s.predict(test_data)
    """

    def __init__(self, settings):
        self.settings = settings
        self.data = settings['data']
        self.nulls = settings['nulls']
        self.epochs = settings['epochs']
        self.vocab = settings['vocab']
        self.model_name = settings['model_name']


    def preprocessing(self):
        data = self.data
        nulls = self.nulls
        lst = []

        for index, text in data.iterrows():
            for word in text['review'].split():
                lst.append(word)
        df = pd.DataFrame(lst)
        df['label'] = 0
        df.rename(columns={0:"word"}, inplace = True)
        df.drop_duplicates(inplace = True)
        df=df.reset_index(drop=True)

        new_nulls = [np.random.choice(nulls) + np.random.choice(nulls) for i in range(150000)]
        df2= pd.DataFrame(new_nulls)
        df2.rename(columns={0:"word"}, inplace = True)
        df2['label'] = 1

        df = df.append(df2)
        df = df.sample(frac=1).reset_index(drop=True)
        self.train_data = df
        return df.head(), df.shape


    def train(self, train_data=None):
        df = self.train_data
        epochs = self.epochs
        encoded_docs = [one_hot(word, vocab) for word in df['word']]
        max_length = 1
        emb_dim = 8

        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

        # define the model
        model = Sequential()
        model.add(Embedding(vocab, emb_dim, input_length=max_length))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # summarize the model
        print(model.summary())
        # fit the model
        model.fit(padded_docs, df['label'], epochs=epochs, verbose=1)
        # evaluate the model
        loss, accuracy = model.evaluate(padded_docs, df['label'], verbose=1)
        print('Accuracy: %f' % (accuracy*100))
        self.model = model


    def save_model(self, model_name=None):
        model_name = model_name if model_name else self.model_name
        model = self.model
        model.save(model_name)
        print('Saved')

    def load_model(self, model_name=None):
        model_name = model_name if model_name else self.model_name
        model = load_model(model_name)
        self.model = model
        print('Model ', model_name, ' loaded')
        return model

    def predict(self, data):
        """
        Predict method processes the input test data and produces a dataframe where all types of null values an 
        if a np.nan format so they can be cleaned with just a simple dropna() method.
        """
        model = self.model

        for col in data.columns:
            data[col] = data[col].astype(str)
        cols = data.columns
        data = data.values

        dftest = pd.DataFrame()

        for line in data:
            initial = line
            line = [one_hot(word, vocab) for word in line]
            line = pad_sequences(line, maxlen=1, padding='post')
            preds = model.predict(np.array(line))
            preds = np.round_(preds)
            preds = np.reshape(preds, (preds.shape[1],preds.shape[0]))
            preds = preds.flatten()
            initial[np.where(preds>0)] = np.nan

        final = pd.DataFrame(data, columns=cols)
        return final
