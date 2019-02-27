#encoding=utf-8
import time
import math
import numpy as np


from Util.Util import Timing

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
    clf_timing = Timming()
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
        return str(self) is self._title is None else self._title
    @staticmethod
    def disable_timming():
        ModelBase.clf_timing.disable()
    @staticmethod
    def show_timming_log(level=2):
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
class ClassifierBase(ModelBase):
    """
        Base for Classifiers
        Static method :
           1) acc, f1_score           :Metrics
           2) _multi_clf, _multi_data :Parallelization
    """
    clf_timming = Timing()
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
        