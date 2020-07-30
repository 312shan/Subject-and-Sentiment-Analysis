from keras import backend as K
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from capsule import *
import jieba

# 这个配置在 windows 中不可用
# jieba.enable_parallel(4)

K.clear_session()

remove_stop_words = False

train_file = '../train.csv'
test_file = '../test_public.csv'

# load stopwords
with open('../hlp_stop_words.txt', encoding='gbk') as f:
    stop_words = set([l.strip() for l in f])

# load Glove Vectors
embeddings_index = {}
EMBEDDING_DIM = 300
embfile = '../word_emb/sgns.baidubaike.bigram-char'
with open(embfile, encoding='utf-8') as f:
    _ = f.readline() # 忽略第一行统计信息
    for i, line in enumerate(f):
        values = line.split()
        words = values[:-EMBEDDING_DIM]
        word = ''.join(words)
        coefs = np.asarray(values[-EMBEDDING_DIM:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

train_df = pd.read_csv(train_file, encoding='utf-8')
test_df = pd.read_csv(test_file, encoding='utf-8')

train_df['label'] = train_df['subject'].str.cat(train_df['sentiment_value'].astype(str))

if remove_stop_words:
    train_df['content'] = train_df.content.map(
        lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
    test_df['content'] = test_df.content.map(
        lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
else:
    train_df['content'] = train_df.content.map(lambda x: ''.join(x.strip().split()))
    test_df['content'] = test_df.content.map(lambda x: ''.join(x.strip().split()))

train_dict = {}
for ind, row in train_df.iterrows():
    content, label = row['content'], row['label']
    if train_dict.get(content) is None:
        train_dict[content] = set([label])
    else:
        train_dict[content].add(label)

conts = []
labels = []
for k, v in train_dict.items():
    conts.append(k)
    labels.append(v)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(labels)

content_list = [jieba.lcut(str(c)) for c in conts]

test_content_list = [jieba.lcut(c) for c in test_df.content.astype(str).values]
word_set = set([word for row in list(content_list) + list(test_content_list) for word in row])
print(len(word_set))
word2index = {w: i + 1 for i, w in enumerate(word_set)}
seqs = [[word2index[w] for w in l] for l in content_list]
seqs_dev = [[word2index[w] for w in l] for l in test_content_list]

embedding_matrix = np.zeros((len(word2index) + 1, EMBEDDING_DIM))
for word, i in word2index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_features = len(word_set) + 1


def get_padding_data(maxlen=100):
    x_train = sequence.pad_sequences(seqs, maxlen=maxlen)
    x_dev = sequence.pad_sequences(seqs_dev, maxlen=maxlen)
    return x_train, x_dev


def get_capsule_model():
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(30, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


maxlen = 100
X_train, X_dev = get_padding_data(maxlen)
print(X_train.shape, X_dev.shape, y_train.shape)

# train model and find params
# model = get_capsule_model()
# batch_size = 30
# epochs = 50
# file_path = "weights_base.best.hdf5"
# checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
# callbacks_list = [checkpoint, early]  # early
# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

first_model_results = []
for i in range(5):
    model = get_capsule_model()
    model.fit(X_train, y_train, batch_size=64, epochs=15)
    first_model_results.append(model.predict(X_dev, batch_size=1024))
pred4 = np.average(first_model_results, axis=0)

tmp = [[i for i in row] for row in pred4]

for i, v in enumerate(tmp):
    if max(v) < 0.5:
        max_val = max(v)
        tmp[i] = [1 if j == max_val else 0 for j in v]
    else:
        tmp[i] = [int(round(j)) for j in v]

tmp = np.asanyarray(tmp)
res = mlb.inverse_transform(tmp)

cids = []
subjs = []
sent_vals = []
for c, r in zip(test_df.content_id, res):
    for t in r:
        if '-' in t:
            sent_val = -1
            subj = t[:-2]
        else:
            sent_val = int(t[-1])
            subj = t[:-1]
        cids.append(c)
        subjs.append(subj)
        sent_vals.append(sent_val)

res_df = pd.DataFrame({'content_id': cids, 'subject': subjs, 'sentiment_value': sent_vals,
                       'sentiment_word': ['一般' for i in range(len(cids))]})

columns = ['content_id', 'subject', 'sentiment_value', 'sentiment_word']
res_df = res_df.reindex(columns=columns)
res_df.to_csv('submit_capsule_word.csv', encoding='utf-8', index=False)


