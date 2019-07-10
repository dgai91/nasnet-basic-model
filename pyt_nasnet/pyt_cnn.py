from torch import nn


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN(nn.Module):

    def __init__(self, num_input, num_class, cnn_config):
        super(CNN, self).__init__()
        cnn_ksize = [c[0] for c in cnn_config]
        num_filters = [c[1] for c in cnn_config]
        max_pool_ksize = [c[2] for c in cnn_config]
        cnn_drop_rate = [c[3] for c in cnn_config]

        self.model = nn.Sequential()
        ic = 1
        feathers = num_input
        for idd, filter_size in enumerate(cnn_ksize):
            conv = nn.Conv1d(in_channels=ic,
                             out_channels=num_filters[idd],
                             kernel_size=(int(filter_size)),
                             stride=1)
            relu = nn.ReLU()
            pool = nn.MaxPool1d(kernel_size=int(max_pool_ksize[idd]),
                                stride=1)

            if cnn_drop_rate[idd] >= 1:
                dropout = nn.Dropout(0.0)
            else:
                dropout = nn.Dropout(cnn_drop_rate[idd])
            self.model.add_module('conv_out_' + str(idd), conv)
            self.model.add_module('relu_' + str(idd), relu)
            self.model.add_module('max_pool_' + str(idd), pool)
            self.model.add_module('dropout_' + str(idd), dropout)
            ic = num_filters[idd]
            feathers = feathers - int(filter_size) + 1 - int(max_pool_ksize[idd]) + 1
        self.model.add_module('flatten', Flatten())
        self.model.add_module('logits', nn.Linear(ic * feathers, num_class))
        self.model.add_module('prediction', nn.Softmax(dim=-1))

    def forward(self, x, **kwargs):
        return self.model(x)
