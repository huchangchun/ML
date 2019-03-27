#encoding=utf-8
import time
import math
import numpy as np


from Util.Timing import Timing

class TimingBase:
    def show_timing_log(self):
        pass
    
class ModelBase:
    """
         Base for all models
         Magic methods:
             1) __str__     : return self.name; __repr__ = __str__
             2) __getitem__ : access to protected members
         Properties:
             1) name  : name of this model, self.__class__.__name__ or self._name
             2) title : used in matplotlib (plt.title())
         Static method:
             1) disable_timing  : disable Timing()
             2) show_timing_log : show Timing() records
     """
    clf_timing = Timing()
    def __init__(self, **kwargs):
        self._plot_label_dict = {}
        self._title = self._name = None
        self._metrics, self._available_metrics = [],{"acc": ClassifierBase.acc}
        self._params = {"sample_weight": kwargs.get("sample_weight",None)}
        
    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)
    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self,"_"+item)
    @property
    def name(self):
        return self.__class__.__name__ if self._name is None else self._name
    @property
    def title(self):
        return str(self) if self._title is None else self._title
    @staticmethod
    def disable_timing():
        ModelBase.clf_timing.disable()
    @staticmethod
    def show_timing_log(level=2):
        ModelBase.clf_timing.show_timing_log(level)
        
    #Handle animation
    @staticmethod
    def _refresh_animation_params(animation_params):
        animation_params["show"] = animation_params.get("show", False)
        animation_params["mp4"] = animation_params.get("mp4", False)
        animation_params["period"] = animation_params.get("period", 1)
    def _get_animation_params(self, animation_params):
        if animation_params is None:
            animation_params = self._params["animation_params"]
        else:
            ClassifierBase._refresh_animation_params(animation_params)
        show, mp4, period = animation_params["show"], animation_params["mp4"],animation_params["period"]
        return show or mp4,show ,mp4,period, animation_params
    
    def get_2d_plot(self, x, y, padding=1, dense=200, draw_background=False, emphasize=None, extra=None, **kwargs):
        pass    

    # Visualization

    def scatter2d(self, x, y, padding=0.5, title=None):
        #to do 
        
        pass
    def scatter3d(self, x, y, padding=0.1, title=None):
        #to do 
        pass
    def predict(self, x, get_raw_results=False, **kwargs):
        pass
class ClassifierBase(ModelBase):
    """
        Base for Classifiers
        Static method :
           1) acc, f1_score           :Metrics
           2) _multi_clf, _multi_data :Parallelization
    """
    clf_timing = Timing()
    def __init__(self, **kwargs):
        super(ClassifierBase,self).__init__(**kwargs)
        self._params["animation_params"] = kwargs.get("animation_params",{})
        ClassifierBase._refresh_animation_params(self._params["animation_params"])
    
    #Metrics
    
    @staticmethod
    def acc(y, y_pred, weights=None):
        y, y_pred = np.asarray(y), np.asarray(y_pred)
        if weights is not None:
            return np.average((y == y_pred) * weights)
        return np.average(y == y_pred) #average([False,True,True]) = 0.66666
 
    #noinspection PyTypeChecker
    def f1_score(y, y_pred):
        """
        f1=2TP/(2*TP + FN + FP)
        """
        tp = np.sum(y * y_pred)
        if tp == 0:
            return .0
        fp = np.sum((1-y) * y_pred)
        fn = np.sum(y * (1 - y_pred))
        return 2 * tp /( 2*tp + fn + fp)
 
    def _multi_data(self, x, task, kwargs, stack=np.hstack, target="single"):
        if target != "parallel":
            return task((x,self,1))
        n_cores = kwargs.get("n_cores",2)
        n_cores = multiprocessing.cpu_count() if n_cores <= 0 else n_cores
        if n_cores == 1:
            matrix = task((x, self,n_cores))
        else:
            pool = Pool(processes=n_cores)
            batch_size = int(len(x) / n_cores)
            batch_base, batch_data, cursor = [], [], 0
            x_dim = x.shape[1]
            for i in range(n_cores):
                if i == n_cores - 1:
                    batch_data.append(x[cursor:])
                    batch_base.append(multiprocessing.Array(ctypes.c_float, (len(x) - cursor) * x_dim))
                else:
                    batch_data.append(x[cursor:cursor + batch_size])
                    batch_base.append(multiprocessing.Array(ctypes.c_float, batch_size * x_dim))
                cursor += batch_size
            shared_arrays = [
                np.ctypeslib.as_array(shared_base.get_obj()).reshape(-1, x_dim)
                for shared_base in batch_base
            ]
            for i, data in enumerate(batch_data):
                shared_arrays[i][:] = data
            matrix = stack(
                pool.map(task, ((x, self, n_cores) for x in shared_arrays))
            )
        return matrix.astype(np.float32)

    def get_metrics(self, metrics):
        if len(metrics) == 0:
            for metric in self._metrics:
                metric.append(metric)
            return metrics
        for i in range(len(metrics) -1 , -1, -1):
            metric = metrics[i]
            if isinstance(metric, str):
                try:
                    metrics[i] = self._available_metrics[metric]
                except AttributeError:
                    metrics.pop(i)
        return metrics
    
    @clf_timing.timeit(level=1, prefix="[API] ")
    def evaluate(self, x, y, metrics=None, tar=0, prefix="Acc", **kwargs):
        if metrics is None:
            metrics = ["acc"]
        self.get_metrics(metrics)
        logs, y_pred = [], self.predict(x, **kwargs)
        print(x)
        print("y",y)
        print("p",y_pred)
        y = np.asarray(y)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        for metric in metrics:
            logs.append(metric(y, y_pred))
        if isinstance(tar, int):
            print(prefix + ": {:12.8}".format(logs[tar]))
        return logs