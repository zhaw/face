import mxnet as mx
import symbol

def train(patch, scale):
    '''
    patch: [0, 1, 2, 3, 4, le, re, nt, lm, rm]
    global patch No.01234 &
    left/right eye, nose tip, left/right mouth corner.
    scale: [1, 2, 3]
    '''
    patch = str(patch)
    scale = str(scale)
    if patch == '0':
        input_shape = (3, 39, 31)
    elif patch in '1234':
        input_shape = (3, 31, 39)
    else:
        input_shape = (3, 31, 31)
    trainIter = mx.io.ImageRecordIter(
            path_imgrec='tmp/%s_%s_tr.rec'%(patch, scale),
            data_shape=input_shape,
            preprocess_threads=8,
            shuffle=True,
            batch_size=128,
            mean_img='tmp/%s_%s_mean.nd'%(patch, scale)
            )
    valIter = mx.io.ImageRecordIter(
            path_imgrec='tmp/%s_%s_val.rec'%(patch, scale),
            data_shape=input_shape,
            preprocess_threads=8,
            shuffle=True,
            batch_size=128,
            mean_img='tmp/%s_%s_mean.nd'%(patch, scale)
            )
    net = symbol.get_deepid_symbol(n_class=10575)
    model = mx.model.FeedForward(symbol=net, ctx=mx.gpu(), num_epoch=100)
    model.fit(X=trainIter, eval_data=valIter, eval_metric='acc',
            epoch_end_callback=mx.callback.do_checkpoint('model/%s_%s'%(patch, scale)),
            batch_end_callback=mx.callback.Speedometer(128, 50))
