"""Utilities for writing training scripts."""

import dataclasses
import pdb
import signal
import subprocess
import sys
import time
import traceback as tb
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Protocol,
    Sized,
    Literal,
    get_type_hints,
    overload,
    
)

import torch


def flattened_hparam_dict_from_dataclass(
    dataclass: Any, prefix: str | None = None
) -> Dict[str, Any]:
    """Convert a config object in the form of a nested dataclass into a
    flattened dictionary, for use with Tensorboard hparams."""
    assert dataclasses.is_dataclass(dataclass)
    cls = type(dataclass)
    hints = get_type_hints(cls)

    output = {}
    for field in dataclasses.fields(dataclass):
        field_type = hints[field.name]
        value = getattr(dataclass, field.name)
        if dataclasses.is_dataclass(field_type):
            inner = flattened_hparam_dict_from_dataclass(value, prefix=None)
            inner = {".".join([field.name, k]): v for k, v in inner.items()}
            output.update(inner)
        # Cast to type supported by tensorboard hparams.
        elif isinstance(value, (int, float, str, bool, torch.Tensor)):
            output[field.name] = value
        else:
            output[field.name] = str(value)

    if prefix is None:
        return output
    else:
        return {f"{prefix}.{k}": v for k, v in output.items()}


def pdb_safety_net():
    """Attaches a "safety net" for unexpected errors in a Python script.

    When called, PDB will be automatically opened when either (a) the user hits Ctrl+C
    or (b) we encounter an uncaught exception. Helpful for bypassing minor errors,
    diagnosing problems, and rescuing unsaved models.
    """

    # Open PDB on Ctrl+C
    def handler(sig, frame):
        pdb.set_trace()

    signal.signal(signal.SIGINT, handler)

    # Open PDB when we encounter an uncaught exception
    def excepthook(type_, value, traceback):  # pragma: no cover (impossible to test)
        tb.print_exception(type_, value, traceback, limit=100)
        pdb.post_mortem(traceback)

    sys.excepthook = excepthook


class SizedIterable[ContainedType](Iterable[ContainedType], Sized, Protocol):
    """Protocol for objects that define both `__iter__()` and `__len__()` methods.

    This is particularly useful for managing minibatches, which can be iterated over but
    only in order due to multiprocessing/prefetching optimizations, and for which length
    evaluation is useful for tools like `tqdm`."""


@dataclasses.dataclass
class LoopMetrics:
    counter: int
    iterations_per_sec: float
    time_elapsed: float


@overload
def range_with_metrics(stop: int, /) -> SizedIterable[LoopMetrics]: ...


@overload
def range_with_metrics(start: int, stop: int, /) -> SizedIterable[LoopMetrics]: ...


@overload
def range_with_metrics(
    start: int, stop: int, step: int, /
) -> SizedIterable[LoopMetrics]: ...


def range_with_metrics(*args: int) -> SizedIterable[LoopMetrics]:
    """Light wrapper for `fifteen.utils.loop_metric_generator()`, for use in place of
    `range()`. Yields a LoopMetrics object instead of an integer."""
    return _RangeWithMetrics(args=args)


@dataclasses.dataclass
class _RangeWithMetrics:
    args: tuple[int, ...]

    def __iter__(self):
        loop_metrics = loop_metric_generator()
        for counter in range(*self.args):
            yield dataclasses.replace(next(loop_metrics), counter=counter)

    def __len__(self) -> int:
        return len(range(*self.args))


def loop_metric_generator(counter_init: int = 0) -> Generator[LoopMetrics, None, None]:
    """Generator for computing loop metrics.

    Note that the first `iteration_per_sec` metric will be 0.0.

    Example usage:
    ```
    # Note that this is an infinite loop.
    for metric in loop_metric_generator():
        time.sleep(1.0)
        print(metric)
    ```

    or:
    ```
    loop_metrics = loop_metric_generator()
    while True:
        time.sleep(1.0)
        print(next(loop_metrics).iterations_per_sec)
    ```
    """

    counter = counter_init
    del counter_init
    time_start = time.time()
    time_prev = time_start
    while True:
        time_now = time.time()
        yield LoopMetrics(
            counter=counter,
            iterations_per_sec=1.0 / (time_now - time_prev) if counter > 0 else 0.0,
            time_elapsed=time_now - time_start,
        )
        time_prev = time_now
        counter += 1


def get_git_commit_hash(cwd: Path | None = None) -> str:
    """Returns the current Git commit hash."""
    if cwd is None:
        cwd = Path.cwd()
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd.as_posix())
        .decode("ascii")
        .strip()
    )


def get_git_diff(cwd: Path | None = None) -> str:
    """Returns the output of `git diff HEAD`."""
    if cwd is None:
        cwd = Path.cwd()
    return (
        subprocess.check_output(["git", "diff", "HEAD"], cwd=cwd.as_posix())
        .decode("utf-8")
        .strip()
    )

def get_experiment_dir(experiment_name: str, 
                       version: int = 0, 
                       experiment_dir: Path =None) -> Path:
    """Creates a directory to put experiment files in, suffixed with a version
    number. Similar to PyTorch lightning."""

    parent_dir = Path(__file__).absolute().parent if experiment_dir is None else experiment_dir
    
    experiment_dir = (
        parent_dir
        / experiment_name
        / f"v{version}"
    ) 
    
    if experiment_dir.exists():
        return get_experiment_dir(experiment_name, version + 1, parent_dir)
    else:
        return experiment_dir



    
class TrainingMetrics:
    def __init__(self,model_type: Literal["dual", "single"] = "dual"):
        
        if model_type=="dual":
            self.reset_dual()
        else:
            self.reset()

    
    def reset(self):
        self.total_loss = 0
        self.joint_rotmat = 0
        self.rel_canonical_transforms = 0
        self.canonical_root_transforms = 0
        self.contacts = 0
        # self.partner_canonical_self_canonical = 0
        self.skating_loss = 0
        self.count = 0
        self.velocity = 0
        
    def reset_dual(self):
        self.joint_rotmat = torch.zeros(2)
        self.rel_canonical_transforms = torch.zeros(2)
        self.canonical_root_transforms = torch.zeros(2)
        self.contacts = torch.zeros(2)
        # self.partner_canonical_self_canonical = 0
        self.count = 0
        self.total_loss = 0
        self.velocity = torch.zeros(2)
        
    def update_dual_joint_loss(self, log_outputs):
        self.count += 1    
        self.total_loss += log_outputs['total_loss']
        for  idx, person in enumerate(['self', 'partner']):
            self.joint_rotmat[idx] += log_outputs[f'{person}_joints'].item()
            self.rel_canonical_transforms[idx] += log_outputs[f'{person}_rel_canonical_transforms'].item()
            self.canonical_root_transforms[idx] += log_outputs[f'{person}_canonical_root_transforms'].item()
            self.contacts[idx] += log_outputs[f'{person}_contacts'].item()
            self.velocity[idx] += log_outputs[f'{person}_velocity'].item()
        # self.partner_canonical_self_canonical += log_outputs['self_partner_canonical_self_canonical'].item()
        
    def get_averages_dual_joint_loss(self,):
        return {
            "total_loss": self.total_loss / self.count,
            "joint_rotmat": { 'self': self.joint_rotmat[0] / self.count, 'partner': self.joint_rotmat[1] / self.count}  ,
            "rel_canonical_transforms": { 'self': self.rel_canonical_transforms[0] / self.count, 'partner': self.rel_canonical_transforms[1] / self.count},
            "canonical_root_transforms": { 'self': self.canonical_root_transforms[0] / self.count, 'partner': self.canonical_root_transforms[1] / self.count},
            "contacts":{ 'self': self.contacts[0] / self.count, 'partner': self.contacts[1] / self.count},
            # "partner_canonical_self_canonical": self.partner_canonical_self_canonical / self.count,
            'velocity': { 'self': self.velocity[0] / self.count, 'partner': self.velocity[1] / self.count},
        }
    def update_dual_separate_loss(self, log_outputs):
        self.count += 1    
        self.total_loss += log_outputs['total_loss']
        for person in ['self', 'partner']:
            self.joint_rotmat += log_outputs[f'{person}_joints'].item()
            self.rel_canonical_transforms += log_outputs[f'{person}_rel_canonical_transforms'].item()
            self.canonical_root_transforms += log_outputs[f'{person}_canonical_root_transforms'].item()
            self.contacts += log_outputs[f'{person}_contacts'].item()
            self.velocity += log_outputs[f'{person}_velocity'].item()
            # self.partner_canonical_self_canonical += log_outputs[f'{person}_partner_canonical_self_canonical'].item()


      
    def update(self, log_outputs):
      self.count += 1    
      self.total_loss += log_outputs['total_loss']
      # self.joint_rotmat += log_outputs['joint_rotmat']
      # self.rel_canonical_transforms += log_outputs['rel_canonical_transforms']
      # self.canonical_root_transforms += log_outputs['canonical_root_transforms']
      # self.contacts += log_outputs['contacts']
      # self.partner_canonical_self_canonical += log_outputs['partner_canonical_self_canonical']
      # self.skating_loss += log_outputs['skating_loss']

    def get_averages(self):
        # self.count = itr
       
      
        return {
            "total_loss": self.total_loss / self.count,
            "joints": self.joint_rotmat / self.count,
            "rel_canonical_transforms": self.rel_canonical_transforms / self.count,
            "canonical_root_transforms": self.canonical_root_transforms / self.count,
            "contacts": self.contacts / self.count,
            # "partner_canonical_self_canonical": self.partner_canonical_self_canonical / self.count,
            "velocity": self.velocity / self.count,
            # "skating_loss": self.skating_loss / self.count,
          }
