from counting import LineCounter, side

# horizontal line y=100 from x=0 to x=200; with a left of b, points below (larger y) are positive
A, B = (0, 100), (200, 100)


def feed(counter, track_id, points, classes=None):
    classes = classes or ["car"] * len(points)
    return [e for p, c in zip(points, classes) if (e := counter.update(track_id, c, p))]


def test_side_sign():
    assert side(A, B, (50, 150)) > 0
    assert side(A, B, (50, 50)) < 0
    assert side(A, B, (50, 100)) == 0


def test_crossing_down_counted_once_with_direction():
    c = LineCounter(A, B, ("down", "up"))
    assert feed(c, 1, [(50, 80), (50, 95), (50, 105), (50, 120)]) == [("down", "car")]
    assert c.counts == {("down", "car"): 1}


def test_crossing_up():
    c = LineCounter(A, B, ("down", "up"))
    assert feed(c, 1, [(50, 120), (50, 90)]) == [("up", "car")]


def test_jitter_on_the_line_counts_once():
    c = LineCounter(A, B)
    assert len(feed(c, 1, [(50, 98), (50, 102), (50, 99), (50, 101), (50, 97), (50, 103)])) == 1


def test_point_exactly_on_line_does_not_reset_side():
    c = LineCounter(A, B)
    assert len(feed(c, 1, [(50, 90), (50, 100), (50, 110)])) == 1


def test_outside_segment_not_counted():
    c = LineCounter(A, B)
    assert feed(c, 1, [(250, 90), (250, 110)]) == []


def test_first_seen_beyond_line_not_counted():
    c = LineCounter(A, B)
    assert feed(c, 1, [(50, 110), (50, 130)]) == []


def test_gap_in_track_still_counted():
    # track missing for several frames, reappears on the other side
    c = LineCounter(A, B)
    assert len(feed(c, 1, [(50, 60), (50, 140)])) == 1


def test_class_by_majority_vote():
    c = LineCounter(A, B)
    events = feed(c, 1, [(50, 70), (50, 80), (50, 90), (50, 110)], ["truck", "bus", "truck", "bus"])
    assert events == [("neg_to_pos", "truck")]  # tie broken by first seen; 2 truck vs 2 bus


def test_tracks_are_independent():
    c = LineCounter(A, B)
    feed(c, 1, [(50, 90), (50, 110)])
    feed(c, 2, [(150, 90), (150, 110)], ["truck", "truck"])
    assert c.counts == {("neg_to_pos", "car"): 1, ("neg_to_pos", "truck"): 1}
