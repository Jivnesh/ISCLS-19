

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import pandas as pd
import numpy as np
import tempfile
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
path = os.chdir('..')
# Generated models will be store in this path
model_dir = tempfile.mkdtemp(dir='./models')

# This function converts compound data into corrosponding FastText features using gensim library.
def BuildFasttext(size,window,min_n,max_n,epochs):
    name='_'+str(size)+'_'+str(window)+'_'+str(min_n)+'_'+str(max_n)+'_'+str(epochs)+'.csv'
    import numpy as np
    import csv
    try:
        fh = open('./data/fast_text_features/FT_model'+name, 'r')
        print("model is saved earlier")
    except FileNotFoundError:
        from gensim.models import FastText
        filepath = './data/Sanskrit_Corpus.txt' 
        raw=[]  
        with open(filepath) as fp:  
           line = fp.readline()
           cnt = 0
           while line:
               tmp=[]
               for word in line.split():
                    tmp.append(word)
               raw.append(tmp)
               line = fp.readline()
               cnt += 1
        print('No of lines in corpus are '+ str(cnt))
        word_count=0
        for i in range(0,len(raw)):
            word_count=word_count+len(raw[i])
        word_count
        FT_model = FastText(size=size, window=window, min_count=1,seed=5)
        print("Building Vocabulory ...")
        FT_model.build_vocab(sentences=raw)
        print("Training FastText model ...")
        print('For 80 epochs it may take more than 30 minutes...')
        FT_model.train(sentences=raw,total_examples=len(raw),total_words=word_count,min_n=min_n,max_n=max_n, epochs=epochs)
        print("Model generated")
        open('./data/fast_text_features/FT_model'+name, 'a').close()
    ###############################################################
     # Get the embedding matrix for compound words vocabulory        
    try:
        print('Getting embedding matrix for vocubolory ...')
        with open('./data/fast_text_features/Embedding'+name, 'r') as f:
             embedding_matrix = list(csv.reader(f, delimiter=","))
        embedding_matrix = np.array(embedding_matrix, dtype=np.float)
        print("File is already present")
    except FileNotFoundError:
        file_name=open("./data/compound_dic.pickle","rb")
        my_dic = pickle.load(file_name)
        embedding_matrix = np.random.uniform(-1, 1, size=(len(my_dic), size))
        for word, i in my_dic.items():
            embedding_matrix[i]=np.array(FT_model.wv[word])
        embedding_matrix = embedding_matrix.astype(np.float32)
        np.savetxt('./data/fast_text_features/Embedding'+name, embedding_matrix, delimiter=",")
    ###############################################################
     # Getting fast-text features of train test data
    try:
        print('Getting features matrix for train and test data ...')
        
        with open('./data/fast_text_features/x_train'+name, 'r') as f:
             x_train = list(csv.reader(f, delimiter=","))
        x_train = np.array(x_train, dtype=np.float)
        
        with open('./data/fast_text_features/y_train'+name, 'r') as f:
             y_train = list(csv.reader(f, delimiter=","))
        y_train = np.array(y_train, dtype=np.float)
        
        with open('./data/fast_text_features/x_test'+name, 'r') as f:
             x_test = list(csv.reader(f, delimiter=","))
        x_test = np.array(x_test, dtype=np.float)
        
        with open('./data/fast_text_features/y_test'+name, 'r') as f:
             y_test = list(csv.reader(f, delimiter=","))
        y_test = np.array(y_test, dtype=np.float)
        print("File is already present")
    except FileNotFoundError:
        data1 = pd.read_csv('./data/train.csv')
        data2 = pd.read_csv('./data/test.csv')
        x_train = np.empty((len(data1),2*size), float)
        x_test = np.empty((len(data2),2*size), float)
        for i in range(0,len(data1)):
            w1=np.array(FT_model.wv[data1.iloc[i,1]])
            w2=np.array(FT_model.wv[data1.iloc[i,2]])
            x_train[i,:] = np.array( np.concatenate((w1, w2)))
        y_train=np.array(data1.iloc[:,3])
        for i in range(0,len(data2)):
            w1=np.array(FT_model.wv[data2.iloc[i,1]])
            w2=np.array(FT_model.wv[data2.iloc[i,2]])
            x_test[i,:] = np.array(np.concatenate((w1, w2)))
        y_test=np.array(data2.iloc[:,3])
        np.savetxt('./data/fast_text_features/x_train'+name, x_train, delimiter=",")
        np.savetxt('./data/fast_text_features/y_train'+name, y_train, delimiter=",")
        np.savetxt('./data/fast_text_features/x_test'+name, x_test, delimiter=",")
        np.savetxt('./data/fast_text_features/y_test'+name, y_test, delimiter=",")
        
    return embedding_matrix,x_train,y_train.astype(int),x_test,y_test.astype(int)


# These parameteres are obtained after tuning
size=350
window=11
min_n=2
max_n=11
epochs=80
embedding_matrix, x_train, y_train, x_test, y_test = BuildFasttext(size,window,min_n,max_n,epochs)

x_strat, x_dev, y_strat, y_dev = train_test_split(x_train, y_train,test_size=0.20,random_state=0,stratify=y_train)


# ## Classification with Tensorflow

# Mapping words to there dictionary index and getting vector corroponding to input words
file_name=open("./data/compound_dic.pickle","rb")
my_dic = pickle.load(file_name)
train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')
x_train_temp=np.empty([len(train), 2], dtype=int)
x_test_voc=np.empty([len(test), 2], dtype=int)
for i in range(0,len(train)):
    x_train_temp[i]=[my_dic.get(train.iloc[i,1]),my_dic.get(train.iloc[i,2])]
for i in range(0,len(test)):
    x_test_voc[i]=[my_dic.get(test.iloc[i,1]),my_dic.get(test.iloc[i,2])]

from sklearn.model_selection import train_test_split
x_train_voc, x_dev_voc, y_train_voc, y_dev_voc = train_test_split(x_train_temp, y_train,test_size=0.20,random_state=0,stratify=y_train)

# Setting parameters for custom estimators
########################################################################
import tensorflow as tf
my_batch_size=100
my_num_epochs=75
my_steps=(my_num_epochs * x_train_voc.shape[0])/my_batch_size
VOC_size=len(my_dic)
########################################################################
# Functions to pass the data to estimators
def my_train_fn(my_num_epochs):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_train_voc)},
    y=np.array(y_train_voc.ravel()),
    num_epochs=my_num_epochs,
    shuffle=False)
    return train_input_fn

def my_dev_fn():
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_dev_voc)},
    y=np.array(y_dev_voc.ravel()),
    num_epochs=1,
    shuffle=False)
    return test_input_fn

def my_test_fn():
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_test_voc)},
    y=np.array(y_test.ravel()),
    num_epochs=1,
    shuffle=False)
    return test_input_fn

def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embedding_matrix.astype(np.float32)

all_classifiers = {}
def train_and_eval(classifier,my_epochs,my_steps):
    all_classifiers[classifier.model_dir] = classifier
    classifier.train(input_fn=my_train_fn(my_epochs), steps= my_steps)
    eval_results = classifier.evaluate(input_fn=my_dev_fn(), steps= my_steps)
    print('Eval Accuracy is ')
    print(eval_results)
    dev_results = classifier.predict(input_fn=my_dev_fn())
    predictions=[]
    for pred in dev_results:
        predictions.append(pred)
    # print(classification_report(y_dev_voc, pd.DataFrame(predictions)))
    # print(confusion_matrix(y_dev_voc, pd.DataFrame(predictions)))
    print("Attention: This result is on test data")
    test_accu = classifier.evaluate(input_fn=my_test_fn(), steps= my_steps)
    print('Test Accuracy is ')
    print(test_accu)
    test_results = classifier.predict(input_fn=my_test_fn())
    test_predictions=[]
    for pred in test_results:
        test_predictions.append(pred)
    print(classification_report(y_test, pd.DataFrame(test_predictions)))
    print(confusion_matrix(y_test, pd.DataFrame(test_predictions)))



# Building feature matrix to inject estimator shape=(3,)
feature_x = tf.feature_column.numeric_column("x", shape=x_train_voc.shape)
# feature_x = tf.feature_column.numeric_column("x", shape=(x_train_voc.shape[1],))
feature_columns = [feature_x]


# 1) MLP based custom estimator

params = {'embedding_initializer': my_initializer}

def dnn_model_fn(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,params):    # And instance of tf.estimator.ModeKeys, see below
    input_layer = tf.contrib.layers.embed_sequence(ids=features["x"], embed_dim=size,vocab_size=VOC_size,
                                                   initializer=params['embedding_initializer'],
                                                   trainable=True)
    
    training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_emb = tf.layers.dropout(inputs=input_layer, rate=0.2, training=training)
    flat = tf.contrib.layers.flatten(dropout_emb)
    h1 = tf.layers.dense(inputs=flat, units= 500, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=h1, units=4)
    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)
    
    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)
   
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)

        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        accuracy = tf.metrics.accuracy(labels, y_pred_cls)       
        metrics =         {
            "Accuracy": accuracy

        }
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('Train_accuracy', accuracy[1])
  
        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec
print('################################################################')
print('MLP based classifier')
print('################################################################')
print('Processing...')
dnn_classifier = tf.estimator.Estimator(model_fn=dnn_model_fn,params=params,
                                        model_dir=os.path.join(model_dir, 'dnn'),)
train_and_eval(dnn_classifier,my_num_epochs,my_steps)


#2) CNN based custom estimator

params = {'embedding_initializer': my_initializer}
def cnn_model_fn(features, labels, mode, params):
       
    input_layer = tf.contrib.layers.embed_sequence(ids=features["x"], embed_dim=size,vocab_size=VOC_size,
                                                   initializer=params['embedding_initializer'],
                                                   trainable=True)
    #######################################################################################
    training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_emb = tf.layers.dropout(inputs=input_layer, rate=0.2, training=training)
    conv = tf.layers.conv1d( inputs=dropout_emb,filters=150,kernel_size=25,padding="same", activation=tf.nn.relu)
    pool = tf.layers.max_pooling1d(inputs = conv, pool_size = 2, strides = 1)
    flat = tf.contrib.layers.flatten(pool)
    logits = tf.layers.dense(inputs=flat , units=4)
    ######################################################################################
    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)
    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)
   
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)

        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        accuracy = tf.metrics.accuracy(labels, y_pred_cls)
       
        metrics =         {
            "Accuracy": accuracy

        }
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('Train_accuracy', accuracy[1])
  
        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec
print('################################################################')
print('CNN based classifier')
print('################################################################')
print('Processing...')
cnn_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, params=params,
                                        model_dir=os.path.join(model_dir, 'cnn'))
train_and_eval(cnn_classifier,my_num_epochs,my_steps)


# 3) LSTM based custom estimator



params = {'embedding_initializer': my_initializer}
def LSTM_model_fn(features, labels, mode, params):
    input_layer = tf.contrib.layers.embed_sequence(ids=features["x"], embed_dim=size,vocab_size=VOC_size,
                                              initializer=params['embedding_initializer'],
                                              trainable=True)
    #################################################################
    training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_emb = tf.layers.dropout(inputs=input_layer, rate=0.2, training=training)
    # create an LSTM cell of size 100
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
    
    # create the complete LSTM
    _, final_states = tf.nn.dynamic_rnn(lstm_cell, dropout_emb, dtype=tf.float32)
    
    # get the final hidden states of dimensionality [batch_size x sentence_size]
    outputs = final_states.h

    logits = tf.layers.dense(inputs=outputs, units=4)
        # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)
    
    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)
    #################################################################


    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)
    
    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)
   
    if mode == tf.estimator.ModeKeys.PREDICT:

        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)

        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        accuracy=tf.metrics.accuracy(labels, y_pred_cls)
        metrics =         {
            "accuracy": accuracy
    
        }
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('Train_accuracy', accuracy[1])

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec
print('################################################################')
print('LSTM based classifier')
print('################################################################')
print('Processing...')
LSTM_classifier = tf.estimator.Estimator(model_fn=LSTM_model_fn,params=params,
                                          model_dir=os.path.join(model_dir, 'LSTM'))
train_and_eval(LSTM_classifier,my_num_epochs,my_steps)

