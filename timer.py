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

class Timers:
    def __init__(self):
        self.timers = {}
        
    def __call__(self, name:str):
        if name not in self.timers:
            self.timers[name] = _Timer(name)

        return self.timers[name]

    def __contains__(self, name:str):
        return name in self.timers

timers = Timers()

Event = None