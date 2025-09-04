# -*- coding: utf-8 -*-
def write_textgrid(path, intervals, xmin=None, xmax=None, tier_name="segment"):
    if not intervals:
        intervals = [(0.0, 0.0, "")]
    if xmin is None: xmin = min(iv[0] for iv in intervals)
    if xmax is None: xmax = max(iv[1] for iv in intervals)
    def s(v): return f"{v:.6f}"
    header = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        '',
        s(xmin), s(xmax),
        '<exists>',
        '1',
        'IntervalTier',
        tier_name,
        s(xmin), s(xmax),
        str(len(intervals))
    ]
    body = []
    for i,(st,ed,tx) in enumerate(intervals, start=1):
        body += [str(i), s(st), s(ed), tx.replace('"','""')]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header+body))
