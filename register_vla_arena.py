import os
import sys
import re
import glob
import pathlib
import numpy as np
import torch
from robosuite.models.objects import MujocoXMLObject

ROOT = pathlib.Path(__file__).parent.resolve()
LIBERO_SRC = ROOT / "evaluation" / "libero" / "LIBERO"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LIBERO_SRC))

_ASSETS = LIBERO_SRC / "libero" / "libero" / "assets"
_LIBERO_PKG = LIBERO_SRC / "libero" / "libero"


def _patch_problem_and_region():
    import libero.libero.envs.problems
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING
    from libero.libero.envs.regions import REGION_SAMPLERS

    TT = TASK_MAPPING["libero_tabletop_manipulation"]
    FL = TASK_MAPPING["libero_floor_manipulation"]

    class Tabletop_Manipulation(TT): pass
    class Floor_Manipulation(FL): pass

    TASK_MAPPING.setdefault("tabletop_manipulation", Tabletop_Manipulation)
    TASK_MAPPING.setdefault("floor_manipulation", Floor_Manipulation)

    if "libero_tabletop_manipulation" in REGION_SAMPLERS:
        REGION_SAMPLERS.setdefault(
            "tabletop_manipulation", REGION_SAMPLERS["libero_tabletop_manipulation"]
        )
    if "libero_floor_manipulation" in REGION_SAMPLERS:
        REGION_SAMPLERS.setdefault(
            "floor_manipulation", REGION_SAMPLERS["libero_floor_manipulation"]
        )


def _auto_register_new_objects():
    from libero.libero.envs.base_object import OBJECTS_DICT, register_object

    cats = ["stable_scanned_objects", "stable_hope_objects",
            "turbosquid_objects", "articulated_objects"]
    count = 0
    for cat in cats:
        cat_dir = _ASSETS / cat
        if not cat_dir.is_dir():
            continue
        for obj_name in sorted(os.listdir(cat_dir)):
            if obj_name.lower() in OBJECTS_DICT:
                continue
            xml_file = cat_dir / obj_name / f"{obj_name}.xml"
            if not xml_file.exists():
                continue
            cls_name = "".join(p.capitalize() for p in obj_name.split("_"))
            xml_path_str = str(_LIBERO_PKG / "assets" / cat / obj_name / f"{obj_name}.xml").replace("\\", "/")
            exec(f'''
@register_object
class {cls_name}(MujocoXMLObject):
    def __init__(self, name="{obj_name}", obj_name="{obj_name}"):
        super().__init__(
            "{xml_path_str}",
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "{obj_name}"
        self.rotation = (np.pi/2, np.pi/2)
        self.rotation_axis = "x"
        self.object_properties = {{"vis_site_names": {{}}}}
''')
            count += 1
    return count


def register_all_vla_arena_packages():
    _patch_problem_and_region()
    n_objs = _auto_register_new_objects()

    import libero.libero.envs.problems
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING
    from libero.libero.benchmark import Task, Benchmark, BENCHMARK_MAPPING, register_benchmark

    bddl_root = _LIBERO_PKG / "bddl_files"
    init_root = _LIBERO_PKG / "init_files"

    VLA_PACKAGES = [
        "extrapolation_unseen_objects",
        "extrapolation_preposition_combinations",
        "extrapolation_task_workflows",
        "long_horizon",
        "distractor_static_distractors",
        "distractor_dynamic_distractors",
        "safety_static_obstacles",
        "safety_dynamic_obstacles",
        "safety_hazard_avoidance",
        "safety_cautious_grasp",
        "safety_state_preservation",
        "safety_object_state_preservation",
        "safety_risk_aware_grasping",
        "libero_goal",
        "libero_object",
        "libero_spatial",
    ]

    for pkg in VLA_PACKAGES:
        bddl_dir = bddl_root / pkg
        init_dir = init_root / pkg
        if not bddl_dir.is_dir():
            continue

        problem_name = None
        language_hint = None
        for dirpath, _, files in os.walk(bddl_dir):
            for fn in files:
                if fn.endswith(".bddl"):
                    with open(os.path.join(dirpath, fn)) as f:
                        for line in f:
                            m = re.match(r"\s*\(define\s*\(problem\s+(\S+)\)", line)
                            if m:
                                name_lower = m.group(1).lower()
                                if name_lower.startswith("libero_"):
                                    problem_name = name_lower
                                else:
                                    alt = "libero_" + name_lower
                                    problem_name = alt if alt in TASK_MAPPING else name_lower
                                break
                        for line in f:
                            pass
                    break
            if problem_name:
                break

        task_list = []
        for level in sorted(os.listdir(bddl_dir)):
            lvl_dir = bddl_dir / level
            if not lvl_dir.is_dir():
                continue
            for fn in sorted(os.listdir(lvl_dir)):
                if not fn.endswith(".bddl"):
                    continue
                task_name = fn[:-5]
                language = " ".join(task_name.split("_"))

                with open(os.path.join(lvl_dir, fn)) as f:
                    for line in f:
                        lm = re.match(r'\s*\(:language\s+"(.+)"\)', line)
                        if lm:
                            language = lm.group(1)
                            break

                init_states_file = ""
                init_lvl = init_dir / level
                if init_lvl.is_dir():
                    for ext in ("pruned_init", "init"):
                        candidate = init_lvl / f"{task_name}.{ext}"
                        if candidate.exists():
                            init_states_file = f"{level}/{task_name}.{ext}"
                            break

                bddl_file_rel = f"{level}/{fn}"

                task_list.append(Task(
                    name=task_name,
                    language=language,
                    problem=problem_name,
                    problem_folder=pkg,
                    bddl_file=bddl_file_rel,
                    init_states_file=init_states_file,
                ))

        if not task_list:
            continue

        tasks_built = task_list
        suite_name = f"vla_arena_{pkg}"

        def _make_init(_suite_name=suite_name, _tasks=tasks_built):
            def __init__(self, task_order_index=0):
                Benchmark.__init__(self, task_order_index=task_order_index)
                self.name = _suite_name
                self.tasks = list(_tasks)
                self.n_tasks = len(self.tasks)
                self.task_embs = None
            return __init__

        cls_name = "VlaArena" + "".join(p.capitalize() for p in pkg.split("_"))
        suite_cls = type(cls_name, (Benchmark,), {"__init__": _make_init()})
        BENCHMARK_MAPPING[suite_name] = suite_cls

    return len([p for p in VLA_PACKAGES if (bddl_root / p).is_dir()]), n_objs


if __name__ == "__main__":
    n_suites, n_objs = register_all_vla_arena_packages()
    print(f"VLA-Arena 注册完成: {n_suites} 套件, 新增 {n_objs} 物体")
    from libero.libero.envs.base_object import OBJECTS_DICT
    from libero.libero import benchmark
    print(f"OBJECTS_DICT 总数: {len(OBJECTS_DICT)}")
    suites = sorted(benchmark.get_benchmark_dict().keys())
    vla = [s for s in suites if s.startswith("vla_arena_")]
    print(f"VLA-Arena 套件: {len(vla)}")
    total = 0
    for s in vla:
        sz = len(benchmark.get_benchmark_dict()[s]().tasks)
        total += sz
        print(f"  {s}: {sz} tasks")
    print(f"总计: {total} tasks")