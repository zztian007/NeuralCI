import io
import os
import sys
import time
from itertools import cycle
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
from tensorflow.keras import backend as kb
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
from scipy import interp
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import CuDNNLSTM, CuDNNGRU, LSTM, GRU, Bidirectional, MaxPooling1D, \
    GlobalAveragePooling1D

lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from attention_layers import scaled_dot_attention, seq_self_attention, many_to_one_attention_with_context, \
    seq_weighted_attention

"""
Some Configurations
"""
# 设置一些参数
OCCUPY_ALL = False   # occupy all GPU or not
EMBEDDING_DIM = 100  # dimension of pre-trained word vector
MAX_SEQUENCE_LENGTH = 500   # max length of a sentence
CLASS_NUMBER = 3  # 设置分类数
RNN_DIM = 128   # RNN output dimension

NUM_FILTERS = 128  # number of convolution kernel
FILTER_SIZES = [2, 3, 4]  # size of convolution kernel
DROP_OUT = 0.5  # drop out rate

BATCH_SIZE = 128  # 最好是2的倍数
EPOCHES = 100  # 设置轮数
PRINT_PER_BATCH = 100  # print result every xxx batches

PRE_TRAINING = True  # use vectors trained by word2vec or not
dataset_split_ratio = 0.1

SEED = 7
SEED = np.random.seed(SEED)
lr = 1e-3  # learning rate
lr_decay = 0.9  # learning rate decay
clip = 6.0  # gradient clipping threshold
l2_reg_lambda = 0.01  # l2 regularization lambda


def log_config(prefix):
    with open(prefix + '#config', 'w') as f:
        f.write('EMBEDDING_DIM ='+str(EMBEDDING_DIM)+'\n')
        f.write('MAX_SEQUENCE_LENGTH='+str(MAX_SEQUENCE_LENGTH)+'\n')
        f.write('NUM_FILTERS='+str(NUM_FILTERS)+'\n')
        f.write('FILTER_SIZES='+str(FILTER_SIZES)+'\n')
        f.write('DROP_OUT='+str(DROP_OUT)+'\n')
        f.write('BATCH_SIZE='+str(BATCH_SIZE)+'\n')
        f.write('EPOCHES='+str(EPOCHES)+'\n')
        f.write('LEARNING_RATE='+str(lr)+'\n')
        f.write('DATASET_SPLIT_RATIO='+str(dataset_split_ratio)+'\n')
        f.close()


def sess_setup():
    if not OCCUPY_ALL:
        # 不全部占满显存, 按需分配
        config = tf.ConfigProto()
        # 或者直接按固定的比例分配。以下代码会占用所有可使用GPU的40%显存。
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        kb.set_session(sess)


def get_pretrained_w2v_model(dic_file_path):
    """
    1.加载词向量 \n
    load the trained instruction vectors
    """
    return Word2Vec.load(dic_file_path)


def construct_ins_embedding():
    """
    construct a vector matrix from pre-trained word2vec models
    :param ins2vec_model:
    :return:
    """
    vocab_size = len(ins2vec_model.wv.index2word)
    print(vocab_size)

    index = 0
    # 存储所有的词语及其索引
    # 初始化 [word : index]
    word_index = {"PAD": index}
    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于padding补零。
    # 行数为所有单词数+1；比如10000+1；列数为词向量“维度”，比如100。
    if ins2vec_model.vector_size != EMBEDDING_DIM:
        print("W2V vector dimension not equal to the configured dimension")
    embeddings_matrix = np.zeros((vocab_size + 2, ins2vec_model.vector_size))

    # 填充上述的字典和矩阵
    for word in ins2vec_model.wv.index2word:
        index = index + 1
        word_index[word] = index
        embeddings_matrix[index] = ins2vec_model.wv[word]

    # OOV词随机初始化为同一向量
    index = index + 1
    word_index["UNKNOWN"] = index
    embeddings_matrix[index] = np.random.rand(ins2vec_model.vector_size) / 10
    return word_index, embeddings_matrix


class RNNModels(object):
    """
    搭建神经网络
    """

    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid', att_type=''):
        """
        initialize key neural network parameters
        :param maxlen: 序列的最大长度
        :param max_features: vocabulary size
        :param embedding_dims: dimension of the word embedding vector
        :param class_num: the number of target class
        :param last_activation: activation function used in the classification layer
        """
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.att_type = att_type

    def do_attention(self, x):
        '''
        add none or one of the following attention layers
        :param x:
        :param att_type:
        :return:
        '''
        att_layer = None
        if self.att_type == 'scaled_dot':
            att_layer = scaled_dot_attention.ScaledDotProductAttention(name='Attention')
            x = att_layer(x)
            # x = GlobalAveragePooling1D()(x)
            x = GlobalMaxPooling1D()(x)
        elif self.att_type == 'seq_self_attention':
            att_layer = seq_self_attention.SeqSelfAttention(attention_activation='sigmoid')
            x = att_layer(x)
            # x = GlobalAveragePooling1D()(x)
            x = GlobalMaxPooling1D()(x)
        elif self.att_type == 'seq_weighted_attention':
            att_layer = seq_weighted_attention.SeqWeightedAttention()
            x = att_layer(x)
            # x = seq_weighted_attention.SeqWeightedAttention()(x)
        elif self.att_type == 'attention_with_context':
            att_layer = many_to_one_attention_with_context.AttentionWithContext()
            x = att_layer(x)
        return x, att_layer

    def get_bidirectional_cudnn_model(self, pre_embeddings, dp_rate=-1.0, use_lstm=False):
        """
        cudnn provided versions, should be much faster
        :param pre_embeddings:
        :param use_lstm: utilize LSTM or GRU unit
        :return: the model
        """
        # Embedding part can try multichannel as same as origin paper
        embedding_layer = Embedding(self.max_features,  # 字典长度
                                    self.embedding_dims,  # 词向量维度
                                    weights=[pre_embeddings],  # 预训练的词向量
                                    input_length=self.maxlen,  # 每句话的最大长度
                                    trainable=False  # 是否在训练过程中更新词向量
                                    )
        input = Input((self.maxlen,))
        embedding = embedding_layer(input)
        if use_lstm:
            x = Bidirectional(CuDNNLSTM(RNN_DIM, return_sequences=True))(embedding)  # LSTM
        else:
            x = Bidirectional(CuDNNGRU(RNN_DIM, return_sequences=True))(embedding)  # GRU

        # add none or one of the following attention layers
        x, atten_layer = self.do_attention(x)
        fn = kb.function([input], [atten_layer.att_weights])

        if dp_rate > 0:
            # 加dropout层
            x = Dropout(dp_rate)(x)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)

        return model, fn

    def get_cudnn_version_model(self, pre_embeddings, dp_rate=-1.0, use_lstm=False):
        """
        cudnn provided versions, should be much faster
        :param pre_embeddings:
        :param use_lstm: utilize LSTM or GRU unit
        :return: the model
        """
        # Embedding part can try multichannel as same as origin paper
        embedding_layer = Embedding(self.max_features,  # 字典长度
                                    self.embedding_dims,  # 词向量维度
                                    weights=[pre_embeddings],  # 预训练的词向量
                                    input_length=self.maxlen,  # 每句话的最大长度
                                    trainable=False  # 是否在训练过程中更新词向量
                                    )
        input = Input((self.maxlen,))
        embedding = embedding_layer(input)
        if use_lstm:
            x = CuDNNLSTM(RNN_DIM, return_sequences=True)(embedding)  # LSTM
        else:
            x = CuDNNGRU(RNN_DIM, return_sequences=True)(embedding)  # GRU

        # add none or one of the following attention layers
        x, atten_layer = self.do_attention(x)
        fn = kb.function([input], [atten_layer.att_weights])

        if dp_rate > 0:
            # 加dropout层
            x = Dropout(dp_rate)(x)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)

        return model, fn

    def get_cnn_rnn_model(self, pre_embeddings, dp_rate=0.0, use_lstm=False, filter_sizes=[2, 3, 4]):
        """
        first CNN to generate a vector, then apply RNN on the vector
        :param pre_embeddings:
        :param dp_rate: drop out rate
        :param use_lstm: utilize LSTM or GRU unit
        :return: the model
        """
        # Embedding part can try multichannel as same as origin paper
        embedding_layer = Embedding(self.max_features,  # 字典长度
                                    self.embedding_dims,  # 词向量维度
                                    weights=[pre_embeddings],  # 预训练的词向量
                                    input_length=self.maxlen,  # 每句话的最大长度
                                    trainable=False  # 是否在训练过程中更新词向量
                                    )
        input = Input((self.maxlen,))
        embedding = embedding_layer(input)

        # add a convolution layer
        c = Conv1D(NUM_FILTERS, 3, padding='valid', activation='relu')(embedding)
        cc = MaxPooling1D()(c)

        if dp_rate > 0:
            # 加dropout层
            cc = Dropout(dp_rate)(cc)

        if use_lstm:
            x = CuDNNLSTM(RNN_DIM, return_sequences=True)(cc)
        else:
            x = CuDNNGRU(RNN_DIM, return_sequences=True)(cc)

        x, atten_layer = self.do_attention(x)
        fn = kb.function([input], [atten_layer.att_weights])

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)

        return model, fn

    def get_naive_version_model(self, pre_embeddings, dp_rate=0.0, use_lstm=False):
        """
        different ways to construct a model from the common ways shown in keras manual
        Are these two ways in constructing a model equivalent?
        :param pre_embeddings:
        :param dp_rate: drop out rate
        :param use_lstm: utilize LSTM or GRU unit
        :return: the model
        """
        # Embedding part can try multichannel as same as origin paper
        embedding_layer = Embedding(self.max_features,  # 字典长度
                                    self.embedding_dims,  # 词向量维度
                                    weights=[pre_embeddings],  # 预训练的词向量
                                    input_length=self.maxlen,  # 每句话的最大长度
                                    trainable=False  # 是否在训练过程中更新词向量
                                    )
        input = Input((self.maxlen,))
        embedding = embedding_layer(input)

        if use_lstm:
            x = LSTM(RNN_DIM, recurrent_dropout=dp_rate, return_sequences=True)(embedding)  # LSTM
        else:
            x = GRU(RNN_DIM, recurrent_dropout=dp_rate, return_sequences=True)(embedding)  # GRU

        x, atten_layer = self.do_attention(x)
        fn = kb.function([input], [atten_layer.att_weights])

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)

        return model, fn

    def add_attention_layer(self, model):
        '''
        add the attention layer for the keras sequential modeling style
        :param model:
        :return:
        '''
        if self.att_type == 'scaled_dot':
            model.add(scaled_dot_attention.ScaledDotProductAttention(name='Attention'))
            model.add(GlobalAveragePooling1D())
        elif self.att_type == 'seq_self_attention':
            model.add(seq_self_attention.SeqSelfAttention(attention_activation='sigmoid'))
            model.add(GlobalAveragePooling1D())
        elif self.att_type == 'seq_weighted_attention':
            model.add(seq_weighted_attention.SeqWeightedAttention())
        elif self.att_type == 'attention_with_context':
            model.add(many_to_one_attention_with_context.AttentionWithContext())

    def get_bidirectional_model(self, pre_embeddings, dp_rate=0.0, use_lstm=False):
        """
        follow the common model construction step shown in keras manual
        :param pre_embeddings:
        :param dp_rate: drop out rate
        :param use_lstm: utilize LSTM or GRU unit
        :return: the model
        """
        # Embedding part can try multichannel as same as origin paper
        embedding_layer = Embedding(self.max_features,  # 字典长度
                                    self.embedding_dims,  # 词向量维度
                                    weights=[pre_embeddings],  # 预训练的词向量
                                    input_length=self.maxlen,  # 每句话的最大长度
                                    trainable=False  # 是否在训练过程中更新词向量
                                    )
        model = Sequential()
        model.add(embedding_layer)

        if use_lstm:
            model.add(Bidirectional(LSTM(RNN_DIM, recurrent_dropout=dp_rate, return_sequences=True)))
        else:
            model.add(Bidirectional(GRU(RNN_DIM, recurrent_dropout=dp_rate, return_sequences=True)))

        self.add_attention_layer(model)

        # model.add(Dropout(dp_rate))
        model.add(Dense(self.class_num, activation=self.last_activation))

        return model

    def get_multilayer_stateful_model(self, pre_embeddings, dp_rate=0.0, use_lstm=False):
        """
        construct 3-layer stateful LSTM or GRU models
        :param pre_embeddings:
        :param dp_rate: drop out rate
        :param use_lstm: utilize LSTM or GRU unit
        :return: the model
        """
        # Embedding part can try multichannel as same as origin paper
        embedding_layer = Embedding(self.max_features,  # 字典长度
                                    self.embedding_dims,  # 词向量维度
                                    weights=[pre_embeddings],  # 预训练的词向量
                                    input_length=self.maxlen,  # 每句话的最大长度
                                    trainable=False  # 是否在训练过程中更新词向量
                                    )
        model = Sequential()
        model.add(embedding_layer)

        if use_lstm:
            model.add(LSTM(RNN_DIM, return_sequences=True,
                           stateful=True))  # returns a sequence of vectors of dimension RNN_DIM
            model.add(LSTM(RNN_DIM, return_sequences=True,
                           stateful=True))  # returns a sequence of vectors of dimension RNN_DIM
            model.add(
                LSTM(RNN_DIM, recurrent_dropout=dp_rate, return_sequences=True,
                     stateful=True))  # return a single vector of dimension RNN_DIM
        else:
            model.add(GRU(RNN_DIM, return_sequences=True,
                          stateful=True))  # returns a sequence of vectors of dimension RNN_DIM
            model.add(GRU(RNN_DIM, return_sequences=True,
                          stateful=True))  # returns a sequence of vectors of dimension RNN_DIM
            model.add(
                GRU(RNN_DIM, recurrent_dropout=dp_rate, return_sequences=True,
                    stateful=True))  # return a single vector of dimension RNN_DIM

        self.add_attention_layer(model)

        # model.add(Dropout(dp_rate))
        model.add(Dense(self.class_num, activation=self.last_activation))

        return model

    def get_multilayer_model(self, pre_embeddings, dp_rate=0.0, use_lstm=False):
        """
        construct 3-layer LSTM or GRU models
        :param pre_embeddings:
        :param dp_rate: drop out rate
        :param use_lstm: utilize LSTM or GRU unit
        :return: the model
        """
        # Embedding part can try multichannel as same as origin paper
        embedding_layer = Embedding(self.max_features,  # 字典长度
                                    self.embedding_dims,  # 词向量维度
                                    weights=[pre_embeddings],  # 预训练的词向量
                                    input_length=self.maxlen,  # 每句话的最大长度
                                    trainable=False  # 是否在训练过程中更新词向量
                                    )
        model = Sequential()
        model.add(embedding_layer)

        if use_lstm:
            model.add(LSTM(RNN_DIM, return_sequences=True))  # returns a sequence of vectors of dimension RNN_DIM
            model.add(LSTM(RNN_DIM, return_sequences=True))  # returns a sequence of vectors of dimension RNN_DIM
            model.add(LSTM(RNN_DIM, recurrent_dropout=dp_rate,
                           return_sequences=True))  # return a single vector of dimension RNN_DIM
        else:
            model.add(GRU(RNN_DIM, return_sequences=True))  # returns a sequence of vectors of dimension RNN_DIM
            model.add(GRU(RNN_DIM, return_sequences=True))  # returns a sequence of vectors of dimension RNN_DIM
            model.add(GRU(RNN_DIM, recurrent_dropout=dp_rate,
                          return_sequences=True))  # return a single vector of dimension RNN_DIM

        self.add_attention_layer(model)

        # model.add(Dropout(dp_rate))
        model.add(Dense(self.class_num, activation=self.last_activation))

        return model


def prepare_data_with_unique(data_dir, func_rep_stg, unique_funs_file, versions, return_fun_names=False):
    """
    Prepare data for compiler family identification task
    Data preparation for other tasks can be easily implemented with slight modifications to this method
    :param data_dir:
    :param func_rep_stg: the function representation strategy, such as 'RSP#coarse.csv', 'WHOLE#fine.csv', 'EDGE#medium.csv'
    :param unique_funs_file:
    :return:
    """
    # organize as a set the unique functions
    fun_name_list = []
    unique_funs = set()
    with open(unique_funs_file) as f:
        line = f.readline()
        while line:
            unique_funs.add(line)
            line = f.readline()
        f.close()

    fun_rep_file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(func_rep_stg):
                fun_rep_file_list.append(file)
    texts = []
    labels = []
    for fname in fun_rep_file_list:
        # print('Processing '+fname)
        fpath = os.path.join(data_dir, fname)
        compiler_setting_str = fname.split('#')[2]
        compiler_options = compiler_setting_str.split('-')
        compiler_family = str(compiler_options[0])
        compiler_version = str(compiler_options[1])
        compiler_opt_level = str(compiler_options[2])
        if compiler_family not in versions.keys():  # skip irrelevant files
            continue
        if compiler_version not in versions[compiler_family]:    # skip irrelevant files
            continue
        with open(fpath) as f:
            lines = f.readlines()
            indices = []
            for line in lines:
                if line.startswith('>>>'):
                    if line.startswith('>>>Func'):
                        f_label = fname.replace(func_rep_stg, '')+'>'+line.split('&')[1]+'\n'
                        if f_label in unique_funs:
                            texts.append(indices)
                            labels.append(label_maps[compiler_family+'-'+compiler_version+'-'+compiler_opt_level])
                            if return_fun_names:
                                fun_name_list.append(f_label)
                    continue
                # 序号化文本，tokenizer句子，并返回每个句子所对应的词语索引
                indices = []
                line = line.rstrip()
                for word in line.split(' '):
                    try:
                        indices.append(word_index[word])  # 把句子中的词语转化为index
                    except:
                        indices.append(word_index['UNKNOWN'])  # OOV词统一用'UNKNOWN'对应的向量表示
            f.close()
    print('transforming labels to array')
    labels = to_categorical(np.asarray(labels), CLASS_NUMBER)  # 将标签转换为数组形式
    # 使用keras的内置函数padding对齐句子，好处是输出numpy数组，不用自己转化了
    print('padding the sequences')
    padded_data = sequence.pad_sequences(texts, maxlen=MAX_SEQUENCE_LENGTH)
    if return_fun_names:
        return padded_data, labels, fun_name_list
    else:
        return padded_data, labels


def prepare_data(data_dir, func_rep_stg, versions, return_fun_names=False):
    """
    Prepare data for compiler family identification task
    Data preparation for other tasks can be easily implemented with slight modifications to this method
    :param data_dir:
    :param func_rep_stg: the function representation strategy, such as 'RSP#coarse.csv', 'WHOLE#fine.csv', 'EDGE#medium.csv'
    :return:
    """
    fun_name_list = []
    fun_rep_file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(func_rep_stg):
                fun_rep_file_list.append(file)
    texts = []
    labels = []
    for fname in fun_rep_file_list:
        # print('Processing '+fname)
        fpath = os.path.join(data_dir, fname)
        compiler_setting_str = fname.split('#')[2]
        compiler_options = compiler_setting_str.split('-')
        compiler_family = str(compiler_options[0])
        compiler_version = str(compiler_options[1])
        compiler_opt_level = str(compiler_options[2])

        if compiler_family not in versions.keys():  # skip irrelevant files
            continue
        if compiler_version not in versions[compiler_family]:    # skip irrelevant files
            continue
        with open(fpath) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('>>>'):
                    if line.startswith('>>>Func'):
                        f_label = fname.replace(func_rep_stg, '') + '>' + line.split('&')[1] + '\n'
                        if return_fun_names:
                            fun_name_list.append(f_label)
                    continue
                # 序号化文本，tokenizer句子，并返回每个句子所对应的词语索引
                indices = []
                line = line.rstrip()
                for word in line.split(' '):
                    try:
                        indices.append(word_index[word])  # 把句子中的词语转化为index
                    except:
                        indices.append(word_index['UNKNOWN'])  # OOV词统一用'UNKNOWN'对应的向量表示
                texts.append(indices)
                labels.append(label_maps[compiler_family + '-' + compiler_version + '-' + compiler_opt_level])
            f.close()
    print('transforming labels to array')
    labels = to_categorical(np.asarray(labels), CLASS_NUMBER)  # 将标签转换为数组形式
    # 使用keras的内置函数padding对齐句子，好处是输出numpy数组，不用自己转化了
    print('padding the sequences')
    padded_data = sequence.pad_sequences(texts, maxlen=MAX_SEQUENCE_LENGTH)
    if return_fun_names:
        return padded_data, labels, fun_name_list
    else:
        return padded_data, labels


def neural_net_train(model, x_train, y_train, val_split=0.1, validation_data=None, fig_prefix=''):
    """
    训练神经网络
    :return:
    """
    print('Training the model: ' + model_store_path)

    """
    earlystopping and modelcheckpoint
    """
    # Callbacks are passed to the model fit the `callbacks` argument in `fit`,
    # which takes a list of callbacks. You can pass any number of callbacks.
    callbacks_list = [
        # This callback will interrupt training when a monitored quantity has stopped improving.
        EarlyStopping(
            # This callback will monitor the validation accuracy of the model
            monitor='val_acc',
            # Training will be interrupted when the validation accuracy has stopped improving for 3 epochs
            patience=5,
        ),
        # This callback will save the current weights after every epoch
        ModelCheckpoint(
            filepath=model_store_path,  # Path to the destination model file
            # The two arguments below mean that we will not overwrite the
            # model file unless `val_acc` has improved, which
            # allows us to keep the best model every seen during training.
            monitor='val_acc',
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            # This callback will monitor the validation loss of the model
            monitor='val_loss',
            # It will divide the learning by 10 when it gets triggered
            factor=0.1,
            # It will get triggered after the validation loss has stopped improving
            # for at least 10 epochs
            patience=5,
        )
    ]
    # Note that since the callback will be monitoring validation accuracy,
    # we need to pass some `validation_data` to our call to `fit`.
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHES,
              callbacks=callbacks_list,
              validation_split=val_split)

    # acc curve during the training
    fig = plt.figure()
    acc = model.history.history['acc']
    val_acc = model.history.history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, '-ko', label='Train accuracy')
    plt.plot(epochs, val_acc, '-k^', label='Validation accuracy')
    plt.title('Train and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.xlim((1, len(acc)+1))
    plt.xticks(np.arange(1, len(acc)+1, 1))
    fig.savefig(fig_prefix + '#accuracy-curve.eps')
    plt.close(fig)

    # loss curve during the training
    fig = plt.figure()
    plt.plot(epochs, model.history.history['loss'], '-ko')
    plt.plot(epochs, model.history.history['val_loss'], '-k^')
    plt.title('Train and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.xlim((1, len(acc) + 1))
    plt.xticks(np.arange(1, len(acc) + 1, 1))
    fig.savefig(fig_prefix + '#loss-curve.eps')
    plt.close(fig)

    return model


def model_evaluaiton(model, fig_prefix=''):
    """
    预测模型的好坏:多分类
    :param model:
    :param fig_prefix:
    :return:
    """
    model.load_weights(model_store_path)
    pre = model.predict(x_test)

    # 计算每一类的ROC Curve和AUC-ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(CLASS_NUMBER):
        fpr[i], tpr[i], thresholds_ = roc_curve(y_test[:, i], pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), pre.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(CLASS_NUMBER)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(CLASS_NUMBER):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= CLASS_NUMBER
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro (area={0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=lw)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro (area={0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='blue', linestyle=':', linewidth=lw)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'pink', 'chocolate',
                    'seagreen', 'mediumslateblue', 'orangered', 'slategray'])
    for i, color in zip(range(CLASS_NUMBER), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (area={1:0.2f})'
                       ''.format(label_strs[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of compiler family identification')
    plt.legend(loc="lower right")
    # plt.show()
    fig.savefig(fig_prefix + '#ROC-curve.eps')
    plt.close(fig)

    for i in range(len(pre)):
        max_value = max(pre[i])
        for j in range(len(pre[i])):
            if max_value == pre[i][j]:
                pre[i][j] = 1
            else:
                pre[i][j] = 0
    # 生成分类评估报告
    report_str = str(classification_report(y_test, pre, digits=4, target_names=label_strs))
    with open(fig_prefix + '#classification_report.txt', 'w') as f:
        f.write(report_str)
        f.close()
    print(report_str)


def model_evaluaiton_for_explain(model, fig_prefix='', fn=None, top_k=10):
    """
    预测模型的好坏:多分类
    :param model:
    :param fig_prefix:
    :return:
    """
    model.load_weights(model_store_path)
    pre = model.predict(x_test)
    att_weights = fn(x_test)
    upper_bound = len(ins2vec_model.wv.index2word) + 1
    with open(fig_prefix + '#explicable.txt', 'w') as f:
        for (pre_label, ground_label, sample, weights, f_name) in zip(pre, y_test, x_test, att_weights[0], func_names):
            f.write('-------------------------------------------------------\n')
            f.write(f_name)
            f.write(str(pre_label) + ' <-> ' + str(ground_label) + '\n')
            for idx, wt in zip(sample, weights):
                if idx == 0:
                    continue
                else:
                    if idx >= upper_bound:
                        f.write(str(idx) + ' {UNKNOWN} ' + str(wt) + '\n')
                    else:
                        f.write(str(idx) + ' {' + ins2vec_model.wv.index2word[idx - 1] + '} ' + str(wt) + '\n')
            f.write('\n')

            sorted = weights.flatten().argsort()[::-1][0:top_k]
            sorted_val = np.sort(weights.flatten())[::-1][0:top_k]
            f.write('Top 10:\n')
            for (idx, val) in zip(sorted, sorted_val):
                if sample[idx] == 0:
                    ins_str = 'PAD'
                elif sample[idx] >= upper_bound:
                    ins_str = 'UNKNOWN'
                else:
                    ins_str = ins2vec_model.wv.index2word[sample[idx] - 1]
                f.write(str(idx) + ' ' + str(val) + ' ' + ins_str)
                f.write('\n')

    for i in range(len(pre)):
        max_value = max(pre[i])
        for j in range(len(pre[i])):
            if max_value == pre[i][j]:
                pre[i][j] = 1
            else:
                pre[i][j] = 0
    # 生成分类评估报告
    report_str = str(classification_report(y_test, pre, digits=4, target_names=label_strs))
    with open(fig_prefix + '#classification_report_explicable.txt', 'w') as f:
        f.write(report_str)
        f.close()
    print(report_str)


def run_whole_procedure(fig_prefix, model_type, versions, att_type):
    """
    execute the whole procedure
    :param fig_prefix:
    :return:
    """
    global ins2vec_model, word_index, x_test, y_test
    # clear existing tf graph at the start of each iteration
    kb.clear_session()
    # re-setup
    sess_setup()
    # step 1: load pre-trained ins2vec model
    print('loading pre-trained ins2vec model...')
    ins2vec_model = Word2Vec.load(dic_file_path)
    # step 2: construct ins2vec embeddings
    print('constructing pre-trained ins2vec embeddings...')
    word_index, embeddings_matrix = construct_ins_embedding()
    # step 3: construct the textcnn model
    print('setting up the neural model...')
    rnn_models = RNNModels(MAX_SEQUENCE_LENGTH, len(embeddings_matrix), EMBEDDING_DIM, CLASS_NUMBER, 'softmax', att_type=att_type)

    if model_type == 'cudnn_ver':
        model, fn = rnn_models.get_cudnn_version_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'cnn_rnn':
        model, fn = rnn_models.get_cnn_rnn_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'bidirectional_cudnn':
        model, fn = rnn_models.get_bidirectional_cudnn_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'naive':
        model, fn = rnn_models.get_naive_version_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'bidirectional':
        model = rnn_models.get_bidirectional_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'multilayer_stateful':
        model = rnn_models.get_multilayer_stateful_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'multilayer':
        model = rnn_models.get_multilayer_model(embeddings_matrix, DROP_OUT)
    else:
        print('!!!!!!!!!!NO SUCH MODEL: ' + model_type)

    # 搭建神经网络完成后，这一步相当于编译它
    opti = optimizers.Adam()
    model.compile(opti, 'categorical_crossentropy', metrics=['accuracy'])  # 多分类
    # step 4: train the model with available dataset
    print('---------------------Step 4---------------------------')
    if unique_fun_stg == 'Intact':
        texts, labels = prepare_data(corpus_path, path_stg_suffix, versions)
    else:
        texts, labels = prepare_data_with_unique(corpus_path, path_stg_suffix, unique_fun_path, versions)
    print('train-test-data split and shuffle...')
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=dataset_split_ratio, random_state=SEED,
                                                        shuffle=True)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    # train the neural network
    print('training...')
    start = time.clock()
    model = neural_net_train(model, x_train=x_train, y_train=y_train, fig_prefix=fig_prefix)
    end = time.clock()
    print('neural-net training takes: %s seconds' % (end - start))
    # model is got and stored
    # step 5: evaluate the model on test data
    model_evaluaiton(model, fig_prefix)


def explain_with_saliency_map(fig_prefix, model_type, versions, att_type, input_files=None, func_list=None, top_k=10):
    """
    get the prediction result explained with saliency map analysis
    :param model_path:
    :param input_files: list of files (each corresponds to an executable)
    :param func_list: list of functions (names) to be analyzed
    :param top_k:
    :return:
    """
    global ins2vec_model, word_index, x_test, y_test, func_names
    # clear existing tf graph at the start of each iteration
    kb.clear_session()
    # re-setup
    sess_setup()
    # step 1: load pre-trained ins2vec model
    print('loading pre-trained ins2vec model...')
    ins2vec_model = Word2Vec.load(dic_file_path)
    # step 2: construct ins2vec embeddings
    print('constructing pre-trained ins2vec embeddings...')
    word_index, embeddings_matrix = construct_ins_embedding()
    # step 3: construct the textcnn model
    print('setting up the neural model...')
    rnn_models = RNNModels(MAX_SEQUENCE_LENGTH, len(embeddings_matrix), EMBEDDING_DIM, CLASS_NUMBER, 'softmax',
                           att_type=att_type)
    if model_type == 'cudnn_ver':
        model, fn = rnn_models.get_cudnn_version_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'cnn_rnn':
        model, fn = rnn_models.get_cnn_rnn_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'bidirectional_cudnn':
        model, fn = rnn_models.get_bidirectional_cudnn_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'naive':
        model, fn = rnn_models.get_naive_version_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'bidirectional':
        model = rnn_models.get_bidirectional_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'multilayer_stateful':
        model = rnn_models.get_multilayer_stateful_model(embeddings_matrix, DROP_OUT)
    elif model_type == 'multilayer':
        model = rnn_models.get_multilayer_model(embeddings_matrix, DROP_OUT)
    else:
        print('!!!!!!!!!!NO SUCH MODEL: ' + model_type)

    if unique_fun_stg == 'Intact':
        x_test, y_test, func_names = prepare_data(corpus_path, path_stg_suffix, versions, True)
    else:
        x_test, y_test, func_names = prepare_data_with_unique(corpus_path, path_stg_suffix, unique_fun_path, versions, True)

    print(len(x_test), 'test sequences')
    # load the trained model
    print('loading...'+model_store_path +'\n')
    # step 5: evaluate the model on test data
    model_evaluaiton_for_explain(model, fig_prefix, fn, top_k)


if __name__ == '__main__':
    # on IST GPU Server
    corpus_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/PhaseI-OrderedPaths/'
    dic_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/PhaseII-ins2vec/'
    model_store_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/Trained-Models/'
    unique_fun_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/UniqueFunList-Strict/'

    # more functions are considered identical and got removed as the stg varies from 'fine' to 'medium' to 'coarse'
    # That is, the "coarse" stg should indicates the most conservative and lower bound performance of trained models
    # 'Intact' means do not remove any functions
    unique_fun_stgs = ['coarse']
    ins_abs_stgs = ['medium']     # instruction abstraction strategies
    versions = {
        'clang': ['3.8', '5.0'],
        'gcc': ['4.7', '6'],
        'icc': ['19.0.4']
    }
    cp_optimization_options = ['O0', 'O1', 'O2', 'O3']

    model_types = ['bidirectional_cudnn']
    # attention_types = ['attention_with_context', 'scaled_dot', 'seq_self_attention']
    attention_types = ['seq_self_attention']

    label_maps = {}
    i = 0
    label_strs = []

    for cp_family in versions.keys():
        # construct the label map
        for ver in versions[cp_family]:
            for opt in cp_optimization_options:
                label_maps[cp_family + '-' + ver + '-' + opt] = i
                if opt == 'O2':
                    continue
                else:
                    i += 1
                    if opt == 'O3':
                        label_strs.append(cp_family + '-' + ver + '-High')
                    else:
                        label_strs.append(cp_family + '-' + ver + '-' + opt)
    CLASS_NUMBER = i

    hyper_log_prefix = 'allSettings'
    log_config(hyper_log_prefix)

    for i in range(len(ins_abs_stgs)):
        # consisted of the 'function representation strategy # instruction abstraction strategy'
        # For example, 'WHOLE#coarse.csv' specifies the function representation strategy is 'WHOLE' and
        # the 'instruction abstraction strategy' is 'coarse'.
        # For the textCNN model defined in this file, the 'function representation strategy' should always be 'WHOLE'
        path_stg_suffix = 'WHOLE#' + ins_abs_stgs[i] + ".csv"
        for unique_fun_stg in unique_fun_stgs:
            for model_type in model_types:
                for att_type in attention_types:
                    # form: 'model type # attention_type #the task # the instruction abstraction strategy # the unique function extraction strategy'
                    # such as 'cnn_rnn#scaled_dot#cp-family#fine#medium'
                    model_name = model_type + '#' + att_type + '#earlystop-allsettings#' + ins_abs_stgs[
                        i] + '#' + unique_fun_stg + '.h5'
                    corpus_path = corpus_base_path + ins_abs_stgs[i]
                    dic_file_path = dic_base_path + 'ins2vec_' + ins_abs_stgs[i] + '.dic'
                    model_store_path = model_store_base_path + model_name
                    unique_fun_path = unique_fun_base_path + 'WHOLE#' + unique_fun_stg + '.csv'

                    run_whole_procedure(model_name.replace('.h5', ''), model_type, versions, att_type)