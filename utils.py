import torch

class ValueHolder:
    def __init__(self):
        self.val = None

    def store(self, val):
        assert self.val is None
        self.val = val

    def pop(self):
        ret = self.val
        self.val = None
        return ret

    def clear(self):
        self.val = None


def array_1d(a, cls):
    return [cls() for _ in range(a)]


def array_2d(a, b, cls):
    return [[cls() for _ in range(b)] for _ in range(a)]


def array_3d(a, b, c, cls):
    return [[[cls() for _ in range(c)] for _ in range(b)] for _ in range(a)]


def array_4d(a, b, c, d, cls):
    return [[[[cls() for _ in range(d)] for _ in range(c)] for _ in range(b)] for _ in range(a)]

    
def prettyTime(seconds):
    # Calculate hours, minutes, and seconds
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    # Create a human-friendly string
    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours}h")
    if minutes > 0:
        time_parts.append(f"{minutes}m")
    if seconds > 0 or (hours == 0 and minutes == 0):
        time_parts.append(f"{seconds}s")
    
    return ":".join(time_parts)

# Example usage
# time_in_seconds = 3665  # 1 hour, 1 minute, and 5 seconds
# print(human_friendly_time(time_in_seconds))

def integerType(t):
    return t in {
        torch.int8, 
        torch.int16,
        torch.int32,
        torch.int64
    }