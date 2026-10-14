import re

from libero.libero.envs.base_object import OBJECTS_DICT, VISUAL_CHANGE_OBJECTS_DICT

from .hope_objects import *
from .google_scanned_objects import *
from .articulated_objects import *
from .turbosquid_objects import *
from .site_object import SiteObject
from .target_zones import *


import difflib

def _resolve_category(name):
    key = name.lower()
    if key in OBJECTS_DICT:
        return key
    stripped = re.sub(r'^rotated_', '', key)
    stripped = re.sub(r'_[xypi0-9]+$', '', stripped)
    if stripped in OBJECTS_DICT:
        return stripped
    with_n = stripped + "_n"
    if with_n in OBJECTS_DICT:
        return with_n
    if key in OBJECTS_DICT:
        return key
    all_keys = list(OBJECTS_DICT.keys())
    close = difflib.get_close_matches(key, all_keys, n=1, cutoff=0.5)
    if close:
        return close[0]
    return key

def get_object_fn(category_name):
    resolved = _resolve_category(category_name)
    return OBJECTS_DICT[resolved]


def get_object_dict():
    return OBJECTS_DICT