import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from scipy.stats import norm
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.engine.topology import Layer
from keras.optimizers import RMSprop


def hard_sigmoid(x):
    x = (100 * x) + 0.5
    zero = K.tf.convert_to_tensor(0., x.dtype.base_dtype)
    one = K.tf.convert_to_tensor(1., x.dtype.base_dtype)
    x = K.tf.clip_by_value(x, zero, one)
    return x


class RD_layer(Layer):
    # x - c > 0 out = jmp x - c < 0 out = 0
    # relu(x - c)

    def __init__(self, **kwargs):
        super(RD_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.d = self.add_weight(name='d',
                                 shape=(input_shape[1],),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.jmp = self.add_weight(name='jmp',
                                   shape=(input_shape[1], 1),
                                   initializer='glorot_uniform',
                                   trainable=True)
        super(RD_layer, self).build(input_shape)

    def call(self, x, **kwargs):
        x = K.bias_add(x, -self.d)
        return K.dot(hard_sigmoid(x), self.jmp) + x

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


def tilted_loss(q, gt, pred):
    e = K.repeat_elements(gt, len(q), axis=1) - pred
    q = np.array(q)
    return K.mean(K.maximum(e * q, e * (q - 1)), axis=-1)


# def mcycleModel(p, qn):
#     x = Input(shape=(p,))
#     h = Dense(units=10, activation='tanh')(x)
#     h = Dropout(0.2)(h)
#     h = Dense(units=64, activation='sigmoid')(h)
#     h = Dropout(0.2)(h)
#     h = Dense(units=128, activation='sigmoid')(h)
#     h = Dropout(0.2)(h)
#     h = Dense(units=128, activation='sigmoid')(h)
#     h = Dropout(0.2)(h)
#     h = Dense(units=64, activation='sigmoid')(h)
#     h = Dropout(0.2)(h)
#     h = Dense(units=10, activation='sigmoid')(h)
#     h = Dropout(0.2)(h)
#     h = Dense(units=qn)(h)
#     # h = RD_layer()(h)
#     model = Model(inputs=x, outputs=h)
#     return model


def mcycleModel(p, qn):
    x = Input(shape=(p,))
    h = Dense(units=16, activation=hard_sigmoid)(x)
    h = Dense(units=32, activation='sigmoid')(h)
    h = Dropout(0.2)(h)
    h = Dense(units=64, activation='sigmoid')(h)
    h = Dropout(0.2)(h)
    h = Dense(units=128, activation='sigmoid')(h)
    h = Dropout(0.2)(h)
    h = Dense(units=128, activation='sigmoid')(h)
    h = Dropout(0.2)(h)
    h = Dense(units=64, activation='sigmoid')(h)
    h = Dropout(0.2)(h)
    h = Dense(units=32, activation='sigmoid')(h)
    h = Dropout(0.2)(h)
    h = Dense(units=16, activation='sigmoid')(h)
    h = Dropout(0.2)(h)
    h = Dense(units=qn)(h)
    # h = RD_layer(name='RD_layer')(h)
    model = Model(inputs=x, outputs=h)
    return model


if __name__ == '__main__':
    n, p, ticks = 10000, 20, 1001
    X = 2 * np.random.rand(n, p) - 1
    Z = np.random.randn(n)
    A = np.random.randn(n)  # latent var
    W = A + Z  # combined var
    Y = 2 * (X[:, 0] <= 0) * A + (X[:, 0] > 0) * W + (1 + (np.sqrt(3) - 1) * (X[:, 0] > 0)) * np.random.randn(n)
    X = np.concatenate([X, np.expand_dims(Z, 1), np.expand_dims(W, 1)], axis=1)
    X_test = np.zeros((ticks, p + 2))
    xvals = np.linspace(-1, 1, ticks)
    X_test[:, 0] = xvals
    truth = xvals > 0
    # JMP = 0.8
    # n, p, ticks = 2000, 20, 1001
    # X = 2 * np.random.rand(n, p) - 1
    # X_test = np.zeros((ticks, p))
    # xvals = np.linspace(-1, 1, ticks)
    # X_test[:, 0] = xvals
    # # np.savez('best_data_model/data/n5_10_64_128_x_y', x=X, y=Y)
    # t1 = -norm.ppf(0.9) + JMP * (xvals > 0)
    # t2 = JMP * (xvals > 0)
    # t3 = norm.ppf(0.9) + JMP * (xvals > 0)
    #
    # qs = [0.1, 0.5, 0.9]
    # color = ['green', 'red', 'yellow']
    # truth = [t1, t2, t3]
    # ensemble_num = 1
    # Y = np.random.randn(n) + JMP * (X[:, 0] > 0)


    y_test_list = []
    optim = RMSprop()
    # model = mcycleModel(p, len(qs))
    # model.compile(loss=lambda y, f: tilted_loss(qs, y, f), optimizer='rmsprop')
    model = mcycleModel(p + 2, 1)
    model.compile(loss='mse', optimizer='rmsprop')
    print(X.shape, Y.shape, X_test.shape)
    model.fit(X, Y, epochs=200, batch_size=16, validation_split=0.3, verbose=2)

    # Predict the quantile
    y_test = model.predict(X_test)
    plt.plot(X_test[:, 0], y_test)  # plot out this quantile
    plt.plot(X_test[:, 0], truth)  # plot out this quantile
    # for idx, (q, t, c) in enumerate(list(zip(qs, truth, color))):
    #     plt.plot(X_test[:, 0], y_test[:, idx], label=q, color=c)  # plot out this quantile
    #     plt.plot(X_test[:, 0], t, label=q, color=c)  # plot out this quantile

    plt.legend()
    plt.show()
