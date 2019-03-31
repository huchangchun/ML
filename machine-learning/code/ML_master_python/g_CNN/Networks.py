#encoding=utf-8
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import matplotlib.pyplot as plt
from g_CNN.Layers import *
from g_CNN.Optimizers import *
from Util.Timing import Timing
from Util.Bases import TFClassifierBase
from Util.ProgressBar import ProgressBar

class NNVerbose:
    NONE = 0
    EPOCH = 1
    ITER = 1.5
    METRICS = 2
    METRICS_DETAIL = 3
    DETAIL = 4
    DEBUG = 5
    
class NN(TFClassifierBase):
    NNTiming = Timing()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._layers = []
        self._optimizers = None
        self._current_dimension = 0
        self._available_metrics = {
            key: value for key, value in zip(["acc", "f1-score"],[NN.acc,NN.f1_score])
        }
        self._metrics, self._metric_names, self._logs = [],[],{}
        
        self.verbose = 0
        self._layer_factory = LayerFactory()
        self._tf_weights, self._tf_bias = [], []
        self._loss = self._train_step = self._inner_y = None
        
        #dic.get(key,value)如果没有找到key,返回value,否则返回dic[key]
        self._params["lr"] = kwargs.get("lr", 0.001)
        self._params["epoch"] = kwargs.get("epoch",10)
        self._params["optimizer"] = kwargs.get("optimizer", "Adam")
        self._params["batch_size"] = kwargs.get("batch_size", 256)
        self._params["train_rate"] = kwargs.get("train_rate", None)
        self._params["metrics"] = kwargs.get("metrics", None)
        self._params["record_period"] = kwargs.get("record_period", 100)
        self._params["verbose"] = kwargs.get("verbose", 1)
        self._params["preview"] = kwargs.get("preview", True)
   
   
    @NNTiming.timeit(level=1)
    def _get_prediction(self, x, name=None, batch_size=1e6, verbose=None):
        if verbose is None:
            verbose = self.verbose
        single_batch = batch_size / np.prod(x.shape[1:])
        single_batch = int(single_batch)
        if not single_batch:
            single_batch = 1
        if single_batch >= len(x):
            return self._sess.run(self._y_pred, {self._tfx: x})
        epoch = int(len(x)) / single_batch
        if not len(x) % single_batch:
            epoch +=1
        name = "Prediction" if name is None else "Prediction ({})".format(name)   
        sub_bar = ProgressBar(max_value=epoch, name=name, start=False)
        if verbose >= NNVerbose.METRICS:
            sub_bar.start()
        rs = [self._sess.run(self._y_pred, {self._tfx: x[:single_batch]})]
        count = single_batch
        if verbose >= NNVerbose.METRICS:
            sub_bar.update()
        while count < len(x):
            count += single_batch
            if count >= len(x):
                rs.append(self._sess.run(self._y_pred, {self._tfx: x[count - single_batch]}))
            else:
                rs.append(self._sess.run(self._y_pred, {self._tfx: x[count - single_batch: count]}))
            if verbose >= NNVerbose.METRICS:
                sub_bar.update()
        return np.vstack(rs)
            
    @NNTiming.timeit(level=4, prefix="[Private StaticMethod] ")
    def _transfer_x(x):
        if len(x.shape) == 1:
            pass
        if len(x.shape) == 4:
            pass
        return x.astype(np.float32)
    @NNTiming.timeit(level=4)
    def _preview(self):
        if not self._layers:
            rs = "None"
        else:
            rs = (
                "Input  :  {:<10s} - {}\n".format("Dimension", self._layers[0].shape[0]) +
                "\n".join(
                    ["Layer  :  {:<10s} - {}".format(
                        _layer.name, _layer.shape[1]
                    ) for _layer in self._layers[:-1]]
                ) + "\nCost   :  {:<10s}".format(self._layers[-1].name)
            )
        print("\n" + "=" * 30 + "\n" + "Structure\n" + "-" * 30 + "\n" + rs + "\n" + "=" * 30)
        print("Optimizer")
        print("-" * 30)
        print(self._optimizer)
        print("=" * 30)    
    @NNTiming.timeit(level=4)
    def _get_w(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name="w")
         
    @NNTiming.timeit(level=4)
    def _add_params(self, shape, conv_channel=None, fc_shape=None, apply_bias=True):
        if fc_shape is not None:
            pass
        elif conv_channel is not None:
            pass
        else:
            w_shape = shape
            b_shape = shape[1]
        self._tf_weights.append(self._get_w(w_shape))
        if apply_bias:
            self._tf_bias.append(self._get_b(b_shape))
        else:
            pass
        
    @NNTiming.timeit(level=4)
    def _add_layer_(self, layer, *args, **kwargs):
        """
        如果是第一层，查找root层的类对象
        """
        #判断存放层的arr为空且layer是个字符串
        if not self._layers and isinstance(layer, str):
            layer = self._layer_factory.get_root_layer_by_name(layer, *args, **kwargs)
            if layer:
                self.add(layer)
                return
        parent = self._layers[-1]
        if isinstance(layer, str):#如果是非第一层
            layer, shape = self._layer_factory.get_layer_by_name(name, parent, 
                self._current_dimension, *args, **kwargs)
            if shape is None:
                self.add(layer)
        else:
            current, nxt = shape
        if isinstance(layer, SubLayer):
            pass
        else:
            fc_shape, conv_channel, last_layer = None, None, self._layers[-1]
            if isinstance(last_layer, ConvLayer):
                pass
            #追加层级信息，添加当前前的weight,更新当前层的维度
            self._layers.append(layer)
            self._add_params((current, nxt), conv_channel, fc_shape, layer.apply_bias)
            self._current_dimension = nxt
            
    @NNTiming.timeit(level=4, prefix="[API] ")
    def add(self, layer, *args, **kwargs):
        if isinstance(layer, str):
            self._add_layer(layer, *args, **kwargs)
        else:
            if not self._layers:
                self._layers, self._current_dimension = [layer], layer.shape[1] #第一次shape是数据的维度，shape[1]是数据的特征维度
                if isinstance(layer, ConvLayer):#如果是卷积
                    pass
                else:
                    self._add_params(layer.shape, apply_bias=layer.apply_bias)
            else:
                if len(layer.shape) == 2:
                    pass
                else:
                    _current, _next = self._current_dimension, layer.shape[0]
                    layer.shape = (_current, _next)
                self._add_layer_(layer, _current, _next)
                
    @NNTiming.timeit(level=1)
    def _get_rs(self, x, predict=True, idx=-1):
        cache = self._layers[0].activate(x, self._tf_weights[0], self._tf_bias[0], predict)
        idx = idx + 1 if idx >= 0 else len(self._layers) + idx + 1
        for i, layer in enumerate(self._layers[1:idx]):
            if i==len(self._layers) - 2: #if i == 1
                if isinstance(self._layers[-2], ConvLayer):
                    pass
                if self._tf_bias[-1] is not None:
                    return tf.matmul(cache, self._tf_weights[-1]) + self._tf_bias[-1]
                return tf.matmul(cacle, self._tf_weights[-1])
            cache = layer.activate(cache, self._tf_weights[i + 1], self._tf_bias[i + 1], predict)
        return cache
    
    @NNTiming.timeit(level=2)
    def _batch_work(self, i, bar, x_train, y_train, y_train_classes, x_test, y_test, y_test_classes, condition):
        if bar is not None:
            condition = bar.update() and condition
        if condition:
            pass
    
    @NNTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, lr=None, epoch=None, batch_size=None, train_rate=None, optimizer=None, metrics=None, record_period=None, verbose=None, preview=None):
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        if optimizer is None:
            optimizer = self._params["optimizer"]
        if batch_size is None:
            batch_size = self._params["batch_size"]
        if train_rate is None:
            train_rate = self._params["train_rate"]
        if metrics is None:
            metrics = self._params["metrics"]
        if record_period is None:
            record_period = self._params["record_period"]
        if verbose is None:
            verbose = self._params["verbose"]
        if preview is None:
            preview = self._params["preview"]
    
        x = NN._transfer_x(x)
        self.verbose = verbose
        self._optimizers = OptFactory().get_optimizer_by_name(optimizer, lr)
        #None=batch_size的占位符, *很有技巧
        """
        a=np.array([[12,21],[123,2]])
        def get(a):
        print(*a.shape)
        print(a.shape)
      
        get(a)
        2 2
        (2, 2)
        """
        self._tfx = tf.placeholder(tf.float32, shape=[None, *x.shape[1:]])
        self._tfy = tf.placeholder(tf.float32, shape=[None, y.shape[1]])
        
        if train_rate is not None:
            train_rate = float(train_rate)
            train_len = int(len(x) * train_rate)
            shuffle_shuffix = np.random.permutation(int(len(x)))
            x, y = x[shuffle_shuffix], y[shuffle_shuffix]
            x_train, y_train = x[:train_len], y[:train_len]
            x_test, y_test = x[train_len:], y[train_len:]
        else:
            x_train = x_test = x
            y_train = y_test = y
        y_train_classes = np.argmax(y_train, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        if metrics is None:
            metrics = []
        self._metrics = self.get_metrics(metrics)
        self._metric_names = [_m.__name__ for _m in metrics]
        self._logs = {
            name: [[] for _ in range(len(metrics) + 1)] for name in ("Train", "Test")
        }

        bar = ProgressBar(max_value=max(1, epoch // record_period), name="Epoch", start=False)
        if self.verbose >= NNVerbose.EPOCH:
            bar.start()

        if preview:
            self._preview()        
        args = (
            (x_train, y_train, y_train_classes,
             x_test,  y_test,  y_test_classes,
             self.verbose >= NNVerbose.METRICS_DETAIL),
            (None, None, x_train, y_train, y_train_classes, x_test, y_test, y_test_classes,
             self.verbose >= NNVerbose.METRICS)
        )
        
        train_repeat = self._get_train_repeat(x, batch_size)
        with self._sess.as_default() as sess:
            self._y_pred = self._get_rs(self._tfx)
            self._inner_y = self._get_rs(self._tfx, predict=False)
            self._loss = self._layers[-1].calculate(self._tfy, self._inner_y)
            self._train_step = self._optimizers.minimize(self._loss)
            sess.run(tf.global_variables_initializer())
            for counter in range(epoch):
                if self.verbose >= NNVerbose.ITER and counter % record_period == 0:
                    sub_bar = ProgressBar(max_value=train_repeat * record_period - 1, name="Iteration")
                else:
                    sub_bar = None
                self._batch_training(x_train, y_train,batch_size, train_repeat, self._loss, self._train_step,sub_bar, *args[0])
                if (counter + 1) % record_period == 0:
                    self._batch_work(*args[1])
                    if self.verbose >= NNVerbose.EPOCH:
                        bar.update(counter // record_period + 1)
    @NNTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, **kwargs):
        y_pred = self._get_prediction(NN.__transfer_x(x))
        if get_raw_results:
            return y_pred
        return np.argmax(y_pred, axis=1)
