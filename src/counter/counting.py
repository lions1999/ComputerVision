"""Count tracked objects crossing a line segment, per direction and class. No YOLO/OpenCV here."""
from collections import Counter, defaultdict


def side(a, b, p):
    """2D cross product (b - a) x (p - a): >0 one side of line a-b, <0 the other, 0 on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def within_segment(a, b, p):
    """True if p projects between a and b, i.e. not beyond the ends of the segment."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
    return 0 <= t <= 1


class LineCounter:
    def __init__(self, a, b, directions=("neg_to_pos", "pos_to_neg")):
        self.a, self.b = a, b
        self.directions = directions  # names for crossing from the negative / positive side
        self.last_side = {}  # track id -> last non-zero side (+1 / -1)
        self.votes = defaultdict(Counter)  # track id -> how often each class was predicted
        self.counted = set()  # each track id is counted at most once
        self.counts = Counter()  # (direction, class) -> number of crossings

    def update(self, track_id, cls, point):
        """Feed one observation of a track; return (direction, class) if it just crossed, else None."""
        self.votes[track_id][cls] += 1
        s = side(self.a, self.b, point)
        if s == 0:
            return None  # exactly on the line: keep the last known side
        s = 1 if s > 0 else -1
        prev = self.last_side.get(track_id)
        self.last_side[track_id] = s
        if prev is None or prev == s or track_id in self.counted:
            return None
        if not within_segment(self.a, self.b, point):
            return None  # crossed the infinite line outside the segment
        self.counted.add(track_id)
        # majority class over the track so far: one "bus" frame on a lorry does not decide
        event = (self.directions[0] if s > 0 else self.directions[1], self.votes[track_id].most_common(1)[0][0])
        self.counts[event] += 1
        return event
