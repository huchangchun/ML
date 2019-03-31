import sys
import os


root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from f_NN.Layers import *
from f_NN.Optimizers import *
import matplotlib.pyplot as plt
from Util.Bases import ClassifierBase
from Util.ProgressBar import ProgressBar

class NNVerbose:
    NONE = 0
    EPOCH = 1
    METRICS = 2
    METRICS_DETAIL = 3
    DETIL = 4
    DEBUG = 5

class NaiveNN(ClassifierBase):
    NaiveNNTiming = Timing()
    def __init__(self, **kwargs):
        super(NaiveNN, self).__init__(**kwargs)
        
        self._layers, self._weights, self._bias = [],[],[] 
        self._w_optimizer = self._b_optimizer = None
        self._current_dimension = 0
        self._params["lr"] = kwargs.get("lr", 0.001)
        self._params["epoch"] = kwargs.get("epoch", 10)
        self._params["optimizer"] = kwargs.get("optimizer", "Adam")
        
        
    @NaiveNNTiming.timeit(level=4)
    def _add_params(self, shape):
        """
        添加层最主要的是确定w，b的维度
        在输入和第一层神经元L1之间w的维度为(x[1],L1的个数)，b的维度为(L1的个数)
        在神经元之间的wi的维度为(Li-1的个数,Li的个数)，b的维度为(Li的个数)
        在神经元与输出Oi之间的wi的维度为(Li-1的个数,Oi的个数==通常等于类别数)，b的维度(Oi的个数)
        """
        self._weights.append(np.random.randn(*shape))
        self._bias.append(np.zeros((1,shape[1])))
        
        
    @NaiveNNTiming.timeit(level=4)
    def _add_layer(self, layer, *args):
        """
        记录current和nxt的维度
        添加w,b
        更新current
        添加层
        """
        current, nxt = args
        self._add_params((current, nxt))
        self._current_dimension = nxt
        self._layers.append(layer)
        
    @NaiveNNTiming.timeit(level=1)
    def _get_activations(self, x):
        """
        activations []存放每一层的激活值
        先计算出第一层神经元的输出，然后后面的神经元激活值通过上一层的输出经过线性变换和激活函数得出
        """
        activations = [self._layers[0].activate(x, self._weights[0],self._bias[0])]
        for i, layer in enumerate(self._layers[1:]):
            activations.append(layer.activate(activations[-1], self._weights[i + 1], self._bias[i+1]))
        return activations
    
    @NaiveNNTiming.timeit(level=1)
    def _get_prediction(self, x):
        return self._get_activations(x)[-1]
    
    @NaiveNNTiming.timeit(level=4)
    def _init_optimizers(self, optimizer, lr, epoch):
        opt_fac = OptFactory()
        self._w_optimizer = opt_fac.get_optimizer_by_name(optimizer, self._weights, lr, epoch)
        self._b_optimizer = opt_fac.get_optimizer_by_name(optimizer, self._bias, lr, epoch)
    @NaiveNNTiming.timeit(level=1)
    def _opt(self, i, _activation, _delta):
        """
        W_(i-1)' = v^T_(i-1) * delta_(i)
        W_(i-1):   n_(i-1) X n_(i) 
        v^T_(i-1): N X n_(i-1)
        delta_(i) :N X n_(i-1)
        b_(i-1)' = sum(deleta_(i))
        """
        self._weights[i] += self._w_optimizer.run(i, _activation.T.dot(_delta))
        self._bias[i] += self._b_optimizer.run(i, np.sum(_delta, axis=0, keepdims=True))
      
    #API  
    @NaiveNNTiming.timeit(level=1,prefix="[API] ")
    def add(self, layer):
        """
        如果self._layer空，则是输入层，初始化w,b,layer(数据维度,神经元个数)
        当前前赋值给self._layers
        如果是隐藏层，layer维度就需要上一层的维度与当前的层的神经元个数共同决定w,b的维度
        添加隐层
        
        """
        if  not self._layers:
            self._layers,self._current_dimension = [layer], layer.shape[1]
            self._add_params(layer.shape)
        else:# def _add_layer(self, layer, *args):
            nxt = layer.shape[0]
            layer.shape = (self._current_dimension,nxt)
            self._add_layer(layer,self._current_dimension,nxt)

    @NaiveNNTiming.timeit(level=1,prefix="[API] ")
    def fit(self, x, y, lr=None, epoch=None, optimizer=None):
        pass
    
    @NaiveNNTiming.timeit(level=4)
    def predict(self, x, get_raw_results=False, **kwargs):
        y_pred=self._get_prediction(np.atleast_2d(x))
        if get_raw_results:
            return y_pred
        return np.argmax(y_pred, axis=1)
    
    
    
class NN(NaiveNN):
    NNTiming = Timing()
    def __init__(self, **kwargs):
        super(NN, self).__init__(** kwargs)
        self._available_metrics = {
            key: value for key, value in zip(["acc", "f1-score"], [NN.acc, NN.f1_score])
        }
        self._metrics, self._metric_names, self._logs = [], [], {}
        self.verbose = None
        self._params["batch_size"] = kwargs.get("batch_size", 256)
        self._params["train_rate"] = kwargs.get("train_rate", None)
        self._params["metrics"] = kwargs.get("metrics", None)
        self._params["record_period"] = kwargs.get("record_period", 100)
        self._params["verbose"] = kwargs.get("verbose", 1)
        
    @NNTiming.timeit(level=1)
    def _get_prediction(self, x, name=None, batch_size=1e6, verbose=None):
        if verbose is None:
            verbose = self.verbose
        single_batch = batch_size / np.prod(x.shape[1:]) #prod 将(2,)->2
        single_batch = int(single_batch)
        if not single_batch:
            single_batch = 1
        if single_batch >=len(x):
            return self._get_activations(x).pop()#返回pop
        epoch = int(len(x) / single_batch)
        if not len(x) % single_batch:
            epoch += 1
        name = "Prediction" if name is None else "Prediction ({})".format(name)
        sub_bar = ProgressBar(max_value=epoch, name=name, start=False)
        if verbose >= NNVerbose.METRICS:
            sub_bar.start()
        rs, count  = [self._get_prediction(x[:single_batch]).pop()], single_batch
        
        if verbose >= NNVerbose.METRICS:
            sub_bar.update()
        """
        count先加然后判断如果count>len(x)，则训练剩余部分:[ count减去single_batch:]
        如果小于len(x)，则训练count个
        """
        while count < len(x):
            count += single_batch
            if count >= len(x):
                rs.append(self._get_prediction(x[count - single_batch:]).pop())
            else:
                rs.append(self._get_prediction(x[count - single_batch:count]).pop())
            if verbose >= NNVerbose.METRICS:
                sub_bar.update()
        return np.vstack(rs) #
    @NNTiming.timeit(level=4, prefix="[API] ")
    def _preview(self):
        if not self._layers:
            rs = "None"
        else:
            rs = ("Input  :  {:<10s} -> {}\n".format("Dimension",self._layers[0].shape[0]) +
                  "\n".join(["Layer  :  {:<10s} - > {}".format(_layer.name, _layer.shape[1]) for 
                            _layer in self._layers[:-1]]
                            ) + "\nCost   :  {:<10s}".format(self._layers[-1].name)
                  )
        print("=" * 30 + "\nStructure\n" + "-" * 30 + "\n" + "-" * 30 + "\n" + rs + "\n" + "=" * 30)
        print("Optimizer")
        print("-" *30)
        print(self._w_optimizer)
        print("=" *30)
         
    @NNTiming.timeit(level=3)
    def _print_metric_logs(self, data_type):
        print()
        print("=" * 47)
        for i, name in enumerate(self._metric_names):
            print("{:<16s} {:<16s}: {:12.8}".format(
                data_type, name, self._logs[data_type][i][-1]))
        print("{:<16s} {:<16s}: {:12.8}".format(
            data_type, "loss", self._logs[data_type][-1][-1]))
        print("=" * 47)

    
    @NNTiming.timeit(level=2)
    def _append_log(self, x, y, y_classes, name):
        y_pred = self._get_prediction(x, name)
        y_pred_classes = np.argmax(y_pred, axis=1)
        for i, metric in enumerate(self._metrics):
            self._logs[name][i].append(metric(y_classes, y_pred_classes))
        self._logs[name][-1].append(self._layers[-1].calculate(y, y_pred) /
                                    len(y))
    @NNTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, lr=None, epoch=None, batch_size=None, train_rate=None,
            optimizer=None,metrics=None, record_period=None, verbose=None):
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
        self.verbose = verbose
        self._init_optimizers(optimizer, lr, epoch)
        layer_width = len(self._layers)
        self._preview()
        """
        如果train_rate不为空
        并打乱数据
        划分训练和测试集
        """
        if train_rate is not None:
            train_rate = float(train_rate)
            train_len = int(len(x) * train_rate)
            shuffle_suffix = np.random.permutation(len(x))
            x, y = x[shuffle_suffix], y[shuffle_suffix]
            x_train, y_train = x[:train_len], y[:train_len]
            x_test, y_test = x[train_len:], y[train_len:]
        else:
            x_train = x_test = x
            y_train = y_test = y
        """
        y是one_hot形式，每个样本的类别通过每个数组中值最大值得下标作为类别
        axis=1表示横向取最大值
        a = np.array([[0,0,0,1]])
        c = np.argmax(a,axis=1)
        c = array([3], dtype=int64)
        """
        y_train_classes = np.argmax(y_train, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        train_len = len(x_train)
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len > batch_size
        train_repeat = 1 if not do_random_batch else int(train_len / batch_size) + 1
        
        if metrics is None:
            metrics = []
        self._metrics = self.get_metrics(metrics)
        self._metric_names = [_m.__name__ for _m in metrics] #"acc"
        self._logs = {
            name:[[] for _ in range(len(metrics) + 1)] for name in ("train", "test") 
        }
        
        bar = ProgressBar(max_value=max(1, epoch // record_period), name="Epoch", start=False)
        if self.verbose >= NNVerbose.EPOCH:
            bar.start()
        sub_bar = ProgressBar(max_value=train_repeat * record_period - 1, name="Iteration", start=False)
        
        for counter in range(epoch):
            if self.verbose >= NNVerbose.EPOCH and counter % record_period == 0:
                sub_bar.start()
            for _ in range(train_repeat):
                if do_random_batch:
                    """如果随机batch则，在train_len中选择batch_size个样本"""
                    batch = np.random.choice(train_len, batch_size)
                    x_batch, y_batch = x_train[batch], y_train[batch]
                else:
                    x_batch, y_batch = x_train, y_train
                self._w_optimizer.update()
                self._b_optimizer.update()
                activations = self._get_activations(x_batch)
                """
                反向传播 :bp
                先求出第一个局部梯度，然后后面的梯度和前一个梯度有关系 &(i) = &(i+1) * w_i^T . u'(i)
            
                更新w,b :opt
                """
                deltas = [self._layers[-1].bp_first(y_batch, activations[-1])]
                for i in range(-1, -len(activations), -1):#i=-1,-2, step=-1
                    deltas.append(
                    self._layers[i - 1].bp(deltas[-1], self._weights[i], activations[i - 1])
                    )#
                for i in range(layer_width - 1, 0, -1):#i=2,1 ,step=-1
                    self._opt(i, activations[i - 1], deltas[layer_width -i - 1])
                #第一层对应的是输入 
                self._opt(0, x_batch, deltas[-1])#对应的是第一个隐藏层， W_(i-1)' = v^T_(i-1) * delta_(i) delta[0]是损失层，delta[-1]是第一层 
                if self.verbose >= NNVerbose.EPOCH:
                    if sub_bar.update() and self.verbose >= NNVerbose.METRICS_DETAIL:
                        self._append_log(x_train, y_train, y_train_classes, "train")
                        self._append_log(x_test, y_test, y_test_classes, "test")
                        self._print_metric_logs("train")
                        self._print_metric_logs("test")
            if self.verbose >= NNVerbose.EPOCH:
                sub_bar.update()
            if (counter + 1) % record_period == 0:
                self._append_log(x_train, y_train, y_train_classes, "train")
                self._append_log(x_test, y_test, y_test_classes, "test")
                if self.verbose >= NNVerbose.METRICS:
                    self._print_metric_logs("train")
                    self._print_metric_logs("test")
                if self.verbose >= NNVerbose.EPOCH:
                    bar.update(counter // record_period + 1)
                    sub_bar = ProgressBar(max_value=train_repeat * record_period - 1, name="Iteration", start=False)
                    
                    
    def draw_logs(self):
        metrics_log, loss_log = {},{}
        for key, value in sorted(self._logs.items()):
            metrics_log[key],loss_log[key] = value[:-1], value[-1]
        for i, name in enumerate(sorted(self._metric_names)):
            plt.figure()
            plt.title("Metric Type:{}".format(name))
            for key, log in sorted(metrics_log.items()):
                xs = np.arange(len(log[i])) + 1
                plt.plot(xs, log[i], label="Data Type:{}".format(key))
            plt.legend(loc=4)
            plt.show()
            plt.close()
        plt.figure()
        plt.title("Cost")
        for key, loss in sorted(loss_log.items()):
            xs = np.arange(len(loss)) + 1
            plt.plot(xs, loss, label="Data Type:{}".format(key))
        plt.legend()
        plt.show()