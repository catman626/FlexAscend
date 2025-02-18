import time

class _Timer:
    ""
    def __init__(self, name: str):
        self.name = name 
        self.started = False
        self.startTime = None

        self.startTimes = []
        self.stopTimes = []

    def start(self, syncFunc:callable = None):
        assert not self.started, f"timer {self.name} has already been started!"

        if syncFunc:
            syncFunc()

        self.started = True

        self.startTime = time.perf_counter()
        self.startTimes.append(self.startTime)

    def stop(self, syncFunc:callable = None):
        assert self.started, f"timer {self.name} has not been started!"
        
        if syncFunc:
            syncFunc()

        stopTime = time.perf_counter()
        self.stopTimes.append(stopTime)

        self.started = False

    def reset(self):

        self.started = False
        self.startTime = None

        self.startTimes = []
        self.stopTimes = []
        
    def elapsed(self, mode:str)->float:
        assert len(self.startTimes) == len(self.stopTimes)
        
        interval = [st - ed for st, ed in zip(self.stopTimes, self.startTimes)]
        if mode == "sum":
            return sum(interval)
        elif mode == "mean":
            return sum(interval) / len(self.startTimes)
        else:
            raise NotImplementedError(f"unrecognized mode in timers elapsed!")


class Timers:
    def __init__(self):
        self.timers:dict[str, _Timer] = {}
        
    def __call__(self, name:str)->_Timer:
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def __contains__(self, name:str):
        return name in self.timers

    # def display(self):
    #     for k in self.timers.keys():
    #         st = self.timers[k].startTimes
    #         ed = self.timers[k].stopTimes
    #         elapse

timers :Timers = Timers()

Event = None