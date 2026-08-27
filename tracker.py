"""
Minimal greedy IOU tracker: assigns persistent integer IDs to boxes across
video frames by matching each frame's boxes to the previous frame's tracked
boxes via IOU (intersection-over-union), above a threshold. Tracks that go
unmatched for too many consecutive frames are dropped.

This is deliberately the simplest thing that works, not a competitor to
ByteTrack or other learned trackers: for a handful of people in a short,
mostly-static clip, matching boxes frame-to-frame by overlap is enough to
give each person a stable identity, which is what's needed to score
per-person compliance in a multi-person clip (instead of pooling every
detected person in a video into one meaningless average).
"""


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class GreedyIOUTracker:
    """
    Call update(boxes) once per frame with that frame's detection boxes
    ([x0, y0, x1, y1] each); get back a track ID per box, same order and
    length as the input. IDs persist across frames for the same physical
    person as long as their box overlaps their previous box above
    iou_threshold; a track survives up to max_age consecutive missed frames
    (e.g. a brief occlusion) before being dropped.

    Defaults (iou_threshold=0.2, max_age=10) are looser than a first pass
    (0.3, 5): the baseline eval in RESULTS.md found handheld camera shake
    on short phone clips moving a person's box enough between frames to
    drop below 0.3 IOU, fragmenting one person into several tracks --
    worse under grounding-dino-tiny's slightly less stable box
    localization. Looser thresholds tolerate more per-frame box drift
    before starting a new track.
    """

    def __init__(self, iou_threshold=0.2, max_age=10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 0
        self._tracks = {}  # track_id -> {"box": [x0,y0,x1,y1], "age": int}

    def update(self, boxes):
        assigned_ids = [None] * len(boxes)

        # Score every (existing track, new box) pair above threshold, then
        # greedily assign highest-IOU pairs first. O(n*m) is fine here --
        # a handful of tracks and detections per frame, not thousands.
        candidates = []
        for tid, track in self._tracks.items():
            for i, box in enumerate(boxes):
                score = iou(track["box"], box)
                if score >= self.iou_threshold:
                    candidates.append((score, tid, i))
        candidates.sort(reverse=True)

        matched_tracks, matched_boxes = set(), set()
        for score, tid, i in candidates:
            if tid in matched_tracks or i in matched_boxes:
                continue
            assigned_ids[i] = tid
            matched_tracks.add(tid)
            matched_boxes.add(i)
            self._tracks[tid] = {"box": boxes[i], "age": 0}

        # Age out tracks that weren't matched this frame; drop the stale ones.
        for tid in list(self._tracks.keys()):
            if tid not in matched_tracks:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]

        # Anything left unmatched is a newly-appeared person.
        for i, box in enumerate(boxes):
            if assigned_ids[i] is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"box": box, "age": 0}
                assigned_ids[i] = tid

        return assigned_ids
