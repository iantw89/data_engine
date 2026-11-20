"""Flags a person already put on images, carried to the other tasks that show the same images.

One track is a candidate in many questions, and every question shows the same few images of it. A person
who has said "this image is someone else" once should not have to say it again in the next question: the
flag is pre-set there (the person sees it and can undo it), long before a round applies the answer and
takes the image out of the track.

What carries:

* ``mixed``, ``not_person``, ``bad_pose`` - statements about the detection itself: wherever it is shown;
* ``unsure`` - too little of the person is visible: that is as true next to other images;
* ``other_person`` - "not the person the rest of this group shows": only into a group that shows at least
  one image of that rest, because next to other images the statement may not hold.

The latest answer about an image wins; an answer that showed the image without a flag clears it.
"""

from __future__ import annotations

from typing import Iterable

from ..web.validate import flag_groups as group_keys  # reference, candidates and the comparison images


def carried_flags(payload: dict, answered: Iterable[dict], *, skip_task: int | None = None) -> dict[str, dict]:
    """``{detection key: {"exclude", "bad_pose", "task"}}`` for the images of ``payload``'s groups that carry a
    flag from an earlier answer. ``answered``: tasks with an answer, any order."""
    groups = group_keys(payload)
    group_of = {key: gid for gid, keys in groups.items() for key in keys}
    if not group_of:
        return {}
    result: dict[str, dict] = {}
    for task in sorted(answered, key=lambda task: (task.get("answered") or 0, task["id"])):
        answer = task.get("answer") or {}
        if task["id"] == skip_task or not isinstance(answer.get("images"), dict):
            continue
        images = answer["images"]
        theirs = group_keys(task["payload"])
        for gid, keys in theirs.items():
            for key in keys:
                if key not in group_of:
                    continue
                flags = images.get(key) or {}
                exclude, bad_pose = flags.get("exclude"), bool(flags.get("bad_pose"))
                if exclude == "other_person":
                    rest = {other for other in keys if other != key and not (images.get(other) or {}).get("exclude")}
                    if not rest & (set(groups[group_of[key]]) - {key}):
                        continue  # said next to other images: it says nothing about this group
                if exclude or bad_pose:
                    result[key] = {"exclude": exclude, "bad_pose": bad_pose, "task": task["id"]}
                else:
                    result.pop(key, None)  # shown later without a flag
    return result


def flagged_unsure(answered: Iterable[dict]) -> set[str]:
    """Detection keys a person left out as 'unsure' (too little visible): poor images to show again."""
    keys = set()
    for task in answered:
        images = (task.get("answer") or {}).get("images")
        if isinstance(images, dict):
            keys.update(key for key, flags in images.items() if isinstance(flags, dict) and flags.get("exclude") == "unsure")
    return keys
