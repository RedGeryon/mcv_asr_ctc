from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Conv2D, Bidirectional, Dropout
from tensorflow.keras.layers import Lambda, BatchNormalization, MaxPooling2D, Flatten, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def crnn_model(args):
    input = Input(name="fcc_feats", shape=(args["timesteps"], args["num_features"], 1))

    m = Conv2D(64, kernel_size=3, activation="relu", padding="same", name="conv1")(input)
    m = BatchNormalization(axis=3)(m)
    m = MaxPooling2D(pool_size=3, strides=2, name="pool1")(m)
    m = Dropout(args["drop_rate"])(m)

    m = Conv2D(128, kernel_size=3, activation="relu", padding="same", name="conv2")(m)
    m = BatchNormalization(axis=3)(m)
    m = MaxPooling2D(pool_size=2, strides=2, name="pool2")(m)
    m = Dropout(args["drop_rate"])(m)

    m = Conv2D(256, kernel_size=3, activation="relu", padding="same", name="conv3")(m)
    m = BatchNormalization(axis=3)(m)
    m = Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", name="conv4")(m)
    m = BatchNormalization(axis=3)(m)
    m = Dropout(args["drop_rate"])(m)

    m = TimeDistributed(Flatten(), name="timedistr")(m)

    m = Bidirectional(GRU(64, return_sequences=True, implementation=2), name="blstm1")(m)
    m = Dense(64, name="blstm1_out", activation="linear", )(m)
    m = Bidirectional(GRU(64, return_sequences=True, implementation=2), name="blstm2")(m)
    y_pred = Dense(args["max_chars"] + 1, name="blstm2_out", activation="softmax")(m)

    base_model = Model(inputs=input, outputs=y_pred)
    base_model.summary()

    labels = Input(name="labels", shape=[args["max_seq_length"], ],
                   dtype="float32")
    input_length = Input(name="input_length", shape=[1, ], dtype="int64")
    label_length = Input(name="label_length", shape=[1, ], dtype="int64")

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)

    opt = Adam(lr=args["lr"], beta_1=0.9, beta_2=0.999,
               decay=0.01, epsilon=10e-8)

    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=opt)
    return base_model, model
