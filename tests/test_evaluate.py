from evaluate import match


def ev(t, d="down", cls="car"):
    return {"t": t, "cls": cls, "dir": d}


def test_perfect_match():
    assert match([ev(1), ev(5)], [ev(1.2), ev(4.9)], tol=1) == [(0, 0), (1, 1)]


def test_missed_and_extra():
    # gt at 5 has no prediction (FN); prediction at 9 has no gt (FP)
    assert match([ev(1), ev(5)], [ev(1.1), ev(9)], tol=1) == [(0, 0)]


def test_direction_must_agree():
    assert match([ev(1, "down")], [ev(1, "up")], tol=1) == []


def test_same_class_preferred_over_closer_time():
    # A40 at 14.4 s: car and truck cross one frame apart in different lanes; time alone pairs them crosswise
    gt = [ev(14.03, cls="car"), ev(14.50, cls="truck")]
    pred = [ev(14.40, cls="truck"), ev(14.43, cls="car")]
    assert match(gt, pred, tol=1) == [(0, 1), (1, 0)]


def test_misclassified_still_matched():
    assert match([ev(2, cls="truck")], [ev(2.1, cls="car")], tol=1) == [(0, 0)]


def test_one_to_one_closest_first():
    # a double count near one real crossing: only one prediction is matched, the other is a FP
    assert match([ev(2)], [ev(1.5), ev(2.1)], tol=1) == [(0, 1)]
