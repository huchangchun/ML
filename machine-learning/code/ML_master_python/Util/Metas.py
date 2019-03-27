from abc import ABCMeta
from Util.Util import Util
from Util.Timing import Timing

class TimingMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        timing = Timing()
        
        for _name, _value in attr.items():
            if "__" in _name or "timing" in _name or "evaluate" in _name:
                continue
            if Util.callable(_value):
                attr[_name] = timing.timeit(level=2)(_value)
        def show_timing_log(self, level=2):
            getattr(self, name + "Timing").show_timing_log(level)
        attr["show_timing_log"] = show_timing_log
        return type(name,bases, attr)