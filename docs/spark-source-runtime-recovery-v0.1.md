# Spark source-CI installation recovery

On 2026-09-06, the first Spark run for commit
`007196258a8e3dee27f58e6e7584d83f4eeffaa6` failed before pytest:
[run 34022794196](https://github.com/saidi19/lumi-model/actions/runs/34022794196).
Its dependency-install log ends with `OSError: [Errno 28] No space left on device`.
It shows default PyTorch 2.14.0 plus CUDA 13, NVIDIA libraries and Triton being
installed alongside PySpark. The wheel build for PySpark completed, but the
package-install step failed. No original Spark test started; no Spark coverage
PASS is claimed from this run. The failed run and initial plan remain preserved.

The correction changes only the new Spark lane and its new runtime/workflow
guards. It installs `torch==2.14.0+cpu` from the
[official CPU package index](https://download.pytorch.org/whl/cpu/torch/), where the
CPython 3.12 Linux x86-64 wheel was verified present. The subsequent normal
project dev/data and PySpark install also pins that same CPU build, so resolution
cannot replace it with the default CUDA package. No pip cache is restored or
written by this lane; existing workflows and their caches are unchanged.
Nothing is deleted from the user's machine or the old CI run.

The runtime guard requires that exact PyTorch build and `torch.version.cuda` to
be `None` before Java/pytest execution. The Python 3.12, Java 17 and PySpark 3.5.6
requirements remain. Three synthetic negative cases reject GPU-tagged or
mismatched Torch builds, and workflow guards reject removal/reordering of the
CPU installation, restoration of the old cache, or pin/index changes.

The three original test bodies, assertions, nine required passing phase reports,
skip rejection, 12-minute process deadline, 15-minute step limit and 20-minute
job limit are unchanged. This is a new corrective commit after a diagnosed
installation failure, not a blind rerun or retry of a started Spark test pass.
Actual Spark acceptance still requires the new commit's hosted execution.

The corrected local guard selection passed 164 tests and 124 subtests, with
the exact CI Ruff rules passing. The first commit's separate Source checks run
[34022794208](https://github.com/saidi19/lumi-model/actions/runs/34022794208)
completed successfully in all four jobs; only the added Spark installation lane
needed this recovery. No private test or old one-shot operation was repeated.
