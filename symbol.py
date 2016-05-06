import mxnet as mx
import numpy as np

#################################
#
#   Deepid stuff
#
#################################

def get_deepid_symbol(n_class):
    data = mx.sym.Variable("data")
    conv1 = mx.sym.Convolution(data=data, kernel=(4,4), num_filter=20, no_bias=False)
    relu1 = mx.sym.Activation(data=conv1, act_type="relu")
    pool1 = mx.sym.Pooling(data=relu1, kernel=(2,2), stride=(2,2), pool_type="max")
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3,3), num_filter=40, no_bias=False)
    relu2 = mx.sym.Activation(data=conv2, act_type="relu")
    pool2 = mx.sym.Pooling(data=relu2, kernel=(2,2), stride=(2,2), pool_type="max")
    # conv3 should use LSWConv
    conv3 = mx.sym.Convolution(data=pool2, kernel=(3,3), num_filter=60, no_bias=False)
    relu3 = mx.sym.Activation(data=conv3, act_type="relu")
    pool3 = mx.sym.Pooling(data=relu3, kernel=(2,2), stride=(2,2), pool_type="max")
    flatten1 = mx.sym.Flatten(data=pool3)
    fc1 = mx.sym.FullyConnected(data=flatten1, num_hidden=160, no_bias=True) # fc1 and fc2 share bias
    # conv4 should use LCConv
    conv4 = mx.sym.Convolution(data=pool3, kernel=(2,2), num_filter=80, no_bias=False)
    act4 = mx.sym.Activation(data=conv4, act_type="relu")
    flatten2 = mx.sym.Flatten(data=act4)
    fc2 = mx.sym.FullyConnected(data=flatten2, num_hidden=160)
    deepid = fc1 + fc2
    ident = mx.sym.FullyConnected(data=deepid, num_hidden=n_class)
    ident = mx.sym.SoftmaxOutput(data=ident, name='softmax')
    return ident

#################################
#
#   Deepid2 stuff
#
#################################

class RecentM():
    def __init__(self, n, init=1):
        self.count = 0
        self.n = n
        self.write_index = 0
        self.m = np.ones(n)*init

    def write(self, new_m):
        self.m[self.write_index] = new_m
        self.write_index += 1
        if self.write_index == n:
            self.write_index = 0
    
    def read(self):
        return np.min(self.m)


def LSWConv(mx.operator.NumpyOp):
    '''
    Convolutional layer with locally shared weights.
    '''
    pass

def LCConv(mx.operator.NumpyOp):
    '''
    Locally connected layer.
    1x1 filters, but don't share weights.
    '''
    pass

def get_deepid2_executor(input_shape, n_class, batch_size, ctx):
    data = mx.sym.Variable("data")
    conv1 = mx.sym.Convolution(data=data, kernel=(4,4), num_filter=20, no_bias=False)
    relu1 = mx.sym.Activation(data=conv1, act_type="relu")
    pool1 = mx.sym.Pooling(data=relu1, kernel=(2,2), stride=(2,2), pool_type="max")
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3,3), num_filter=40, no_bias=False)
    relu2 = mx.sym.Activation(data=conv2, act_type="relu")
    pool2 = mx.sym.Pooling(data=relu2, kernel=(2,2), stride=(2,2), pool_type="max")
    # conv3 should use LSWConv
    conv3 = mx.sym.Convolution(data=pool2, kernel=(3,3), num_filter=60, no_bias=False)
    relu3 = mx.sym.Activation(data=conv3, act_type="relu")
    pool3 = mx.sym.Pooling(data=relu3, kernel=(2,2), stride=(2,2), pool_type="max")
    flatten1 = mx.sym.Flatten(data=pool3)
    fc1 = mx.sym.FullyConnected(data=flatten1, num_hidden=160, no_bias=True) # fc1 and fc2 share bias
    # conv4 should use LCConv
    conv4 = mx.sym.Convolution(data=pool3, kernel=(2,2), num_filter=80, no_bias=False)
    act4 = mx.sym.Activation(data=conv4, act_type="relu")
    flatten2 = mx.sym.Flatten(data=act4)
    fc2 = mx.sym.FullyConnected(data=flatten2, num_hidden=160)
    deepid = fc1 + fc2
    ident = mx.sym.FullyConnected(data=deepid, num_hidden=n_class)
    ident = mx.sym.SoftmaxActivation(data=ident)
    out = mx.sym.Group([ident, deepid])
    
    arg_shapes, output_shapes, aux_shapes = out.infer_shape(data=(batch_size, 3, input_shape[0], input_shape[1]))
    arg_names = out.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shape]))
    grad_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shape]))
    
    executor = out.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req="write")
    return executor

def verif_loss_grad(f1, f2, y, rm):
    '''
    Input Params:
    f1: deepid1
    f2: deepid2
    y:  1 if f1 f2 are same person, -1 otherwise
    rm: recent margin param
    Output Params:
    g1: grad for f1
    g2: grad for f2
    '''
    if y == 1:
        g1 = f2 - f1
        g2 = f1 - f2
    else:
        l2 = mx.nd.sqrt(mx.nd.sum(mx.nd.square(f1-f2)))
        m = rm.read()
        rm.write(l2)
        if l2 >= m:
            g1 = mx.nd.zeros(f1.shape)
            g2 = mx.nd.zeros(f2.shape)
        else:
            g1 = .5 * (m-l2) / l2 * (f1-f2)
            g2 = .5 * (m-l2) / l2 * (f2-f1)
    return g1, g2, rm
