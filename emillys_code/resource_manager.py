import os
import re
import pathlib
import psutil
from typing import Optional, Tuple, Any

SENTINEL_NO_LIMIT = 1 << 60  # ~1 EiB / sentinel for "unlimited"

# ---------- cgroups detection ----------
def read_cgroup_limit_bytes() -> Optional[int]:
    """
    Read the memory limit (in bytes) imposed by Linux cgroups.

    The function checks for memory constraints in both cgroup v2 and cgroup v1:
    - **cgroup v2**: Reads ``/sys/fs/cgroup/memory.max``. The special value
      ``"max"`` indicates no limit. Otherwise, interprets the file contents
      as an integer number of bytes.
    - **cgroup v1**: Reads ``/sys/fs/cgroup/memory/memory.limit_in_bytes``.
      Values greater than or equal to ``2**60`` or equal to ``0`` are treated
      as "no limit".

    Returns
    -------
    Optional[int]
        The memory limit in bytes, or ``None`` if no finite limit is detected
        or if files cannot be read.

    Notes
    -----
    - This reads *configured* limits only, not actual usage.
    - Failures (missing files, permissions, unparsable values) are silently
      ignored and treated as "no limit".
    """
    # cgroup v2
    try:
        p = pathlib.Path("/sys/fs/cgroup/memory.max")
        if p.exists():
            txt = p.read_text().strip().lower()
            if txt == "max":
                return None
            if txt.isdigit():
                val = int(txt)
                return None if val >= SENTINEL_NO_LIMIT else val
    except Exception:
        pass

    # cgroup v1
    try:
        p = pathlib.Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if p.exists():
            txt = p.read_text().strip().lower()
            if txt.isdigit():
                val = int(txt)
                return None if val >= SENTINEL_NO_LIMIT or val == 0 else val
    except Exception:
        pass

    return None

# ---------- SLURM detection ----------

def _env_int(name: str) -> Optional[int]:
    """
    Extract the first integer from an environment variable.

    Parameters
    ----------
    name : str
        The environment variable name.

    Returns
    -------
    Optional[int]
        The first integer value found in the variable, or ``None`` if the
        variable is unset, empty, or does not contain digits.

    Examples
    --------
    >>> os.environ["SLURM_JOB_CPUS_PER_NODE"] = "16(x2)"
    >>> _env_int("SLURM_JOB_CPUS_PER_NODE")
    16
    """
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    # handle strings like "16(x2)" by grabbing the first integer
    m = re.search(r"\d+", v)
    return int(m.group()) if m else None

def slurm_limit_bytes() -> Optional[int]:
    """
    Infer a SLURM-imposed memory limit in bytes.

    This function inspects common SLURM environment variables in priority order:

    1. ``SLURM_MEM_PER_NODE`` (MiB).
    2. ``SLURM_MEM_PER_CPU`` (MiB) multiplied by an inferred CPU count.
       CPU count is taken from (in order):
       ``SLURM_CPUS_PER_TASK``, ``SLURM_CPUS_ON_NODE``,
       or the first integer in ``SLURM_JOB_CPUS_PER_NODE``.

    Returns
    -------
    Optional[int]
        Memory limit in bytes, or ``None`` if no limit is derivable.

    Notes
    -----
    - Memory values are interpreted as mebibytes (MiB).
    - CPU count parsing is conservative: only the first integer is used from
      strings like ``"16(x2)"``.
    - Returns ``None`` if the derived limit is zero or missing.
    """
    per_node_mib = _env_int("SLURM_MEM_PER_NODE")
    if per_node_mib and per_node_mib > 0:
        return per_node_mib * 1024**2

    per_cpu_mib = _env_int("SLURM_MEM_PER_CPU")
    if per_cpu_mib and per_cpu_mib > 0:
        cpus = (
            _env_int("SLURM_CPUS_PER_TASK")
            or _env_int("SLURM_CPUS_ON_NODE")
            or _env_int("SLURM_JOB_CPUS_PER_NODE")
        )
        if cpus and cpus > 0:
            return per_cpu_mib * cpus * 1024**2

    return None

# ---------- Budget resolution ----------
def resolve_ram_budget_bytes(user_budget_gb: float) -> int:
    """
    Resolve a conservative RAM budget in bytes.

    Candidates considered:
    - User-specified budget (if > 0), converted GiB → bytes.
    - cgroup memory limit, if any.
    - SLURM memory limit, if any.
    - System physical RAM (from :func:`psutil.virtual_memory`).

    The minimum of all available candidates is returned, ensuring the result
    never exceeds the strictest known cap.

    Parameters
    ----------
    user_budget_gb : float
        User-requested budget in GiB. If ≤ 0, ignored.

    Returns
    -------
    int
        Final hard cap in bytes (always > 0).
    """
    user = int(user_budget_gb * 1024**3) if user_budget_gb and user_budget_gb > 0 else None
    candidates = [
        c for c in (
            user,
            read_cgroup_limit_bytes(),
            slurm_limit_bytes(),
            psutil.virtual_memory().total,
        )
        if isinstance(c, int) and c > 0
    ]
    # psutil ensures at least one candidate
    return min(candidates)

# ---------- n_jobs clamping ----------
def _sanitize_parallel_params(
    n_jobs_req: int,
    gb_per_job: float,
    reserve_frac: float,
    logger: Any,
) -> Tuple[int, float, float, Tuple[int, float, float]]:
    """
    Validate and sanitize parallel job configuration parameters.

    Parameters
    ----------
    n_jobs_req : int
        Requested number of jobs. Values ≤ 0 are replaced by 1 with a warning.
    gb_per_job : float
        Estimated memory per job (GiB). Values ≤ 0 are replaced by 0.1 GiB
        with a warning.
    reserve_frac : float
        Fraction of total budget to reserve for system overhead. Values
        outside [0, 1) are clamped into [0.0, 0.99] with a warning.
    logger : Any
        Logger object providing ``warning`` method.

    Returns
    -------
    tuple
        (n_jobs_req, gb_per_job, reserve_frac, orig) where:
        - n_jobs_req : int (sanitized number of jobs)
        - gb_per_job : float (sanitized memory per job in GiB)
        - reserve_frac : float (sanitized reserve fraction)
        - orig : tuple of the original input values
    """
    orig = (n_jobs_req, gb_per_job, reserve_frac)

    if n_jobs_req is None or n_jobs_req <= 0:
        logger.warning("Requested --n-jobs=%s is invalid; setting to 1.", n_jobs_req)
        n_jobs_req = 1

    if gb_per_job is None or gb_per_job <= 0:
        logger.warning("gb_per_job %.3f must be > 0; setting to 0.1.", gb_per_job if gb_per_job is not None else float("nan"))
        gb_per_job = 0.1

    if reserve_frac is None or not (0.0 <= reserve_frac < 1.0):
        logger.warning("reserve_frac %s is out of [0,1); clamping into [0.0, 0.99].", reserve_frac)
        reserve_frac = min(max(reserve_frac or 0.0, 0.0), 0.99)

    return n_jobs_req, gb_per_job, reserve_frac, orig

def clamp_n_jobs_by_budget(
    n_jobs_req: int,
    gb_per_job: float,
    budget_bytes: int,
    reserve_frac: float,
    logger: Any,
) -> int:
    """
    Clamp requested jobs to respect memory budget constraints.

    Parameters
    ----------
    n_jobs_req : int
        Number of requested parallel jobs (≥ 1).
    gb_per_job : float
        Estimated memory per job in GiB (> 0).
    budget_bytes : int
        Hard memory budget in bytes.
    reserve_frac : float
        Fraction of the budget reserved for system overhead (0 ≤ reserve_frac < 1).
    logger : Any
        Logger object providing ``warning`` method.

    Returns
    -------
    int
        Clamped number of jobs (≥ 1).

    Notes
    -----
    - Usable memory = ``budget_bytes × (1 − reserve_frac)``.
    - Maximum feasible jobs = floor(usable / per_job).
    - If requested jobs exceed feasible jobs, clamps down and logs a warning.
    """
    usable = int(budget_bytes * (1.0 - reserve_frac))  # leave headroom
    per_job_bytes = max(1, int(gb_per_job * 1024**3))  # avoid div/0
    max_jobs = max(1, usable // per_job_bytes)

    if n_jobs_req > max_jobs:
        logger.warning(
            "Clamping --n-jobs from %d to %d to respect RAM budget: usable≈%.1fGB, per_job≈%.1fGB.",
            n_jobs_req, max_jobs, usable / 1024**3, per_job_bytes / 1024**3,
        )
    return max(1, min(n_jobs_req, max_jobs))

def get_n_jobs_config(logger: Any, args: Any) -> int:
    """
    Compute a safe `n_jobs` under memory constraints.

    This function:
    1. Sanitizes input parameters from ``args``.
    2. Resolves the effective memory budget (bytes).
    3. Logs system total RAM, requested budget, effective budget, reserve,
       and per-job estimate.
    4. Warns if even one job is likely to exceed the usable budget.
    5. Clamps ``n_jobs`` if needed.

    Parameters
    ----------
    logger : Any
        Logger providing ``info``, ``warning``, and ``error`` methods.
    args : Any
        Object with attributes:
        - ``ram_budget_gb`` : float
        - ``reserve_frac`` : float
        - ``gb_per_job`` : float
        - ``n_jobs`` : int

    Returns
    -------
    int
        Final safe number of jobs (≥ 1).

    Raises
    ------
    ValueError
        If no valid memory candidates exist (should not occur, since system
        RAM is always available).
    """
    # Sanitize BEFORE logging so logs match the effective values
    n_jobs_req, gb_per_job, reserve_frac, orig = _sanitize_parallel_params(
        getattr(args, "n_jobs", 1),
        getattr(args, "gb_per_job", 1.0),
        getattr(args, "reserve_frac", 0.4),
        logger,
    )

    budget_bytes = resolve_ram_budget_bytes(getattr(args, "ram_budget_gb", 0.0))
    system_total_gb = psutil.virtual_memory().total / 1024**3
    effective_budget_gb = budget_bytes / 1024**3
    usable_gb = effective_budget_gb * (1.0 - reserve_frac)

    logger.info(
        "System RAM total ≈ %.1f GiB; requested --ram-budget-gb=%.1f GiB.",
        system_total_gb, getattr(args, "ram_budget_gb", 0.0),
    )
    logger.info(
        "Effective RAM budget after env limits: ≈ %.1f GiB; usable after reserve(%.0f%%): ≈ %.1f GiB.",
        effective_budget_gb, reserve_frac * 100, usable_gb,
    )
    logger.info("Per-job estimate: %.1f GiB; requested n_jobs=%d.", gb_per_job, n_jobs_req)

    # Early warning if even a single job likely exceeds usable memory
    if gb_per_job > usable_gb:
        logger.error(
            "Per-job estimate (%.1f GiB) exceeds usable memory (%.1f GiB) after reserve. "
            "Consider lowering --gb-per-job or --reserve-frac, or increasing --ram-budget-gb.",
            gb_per_job, usable_gb,
        )

    n_jobs = clamp_n_jobs_by_budget(
        n_jobs_req=n_jobs_req,
        gb_per_job=gb_per_job,
        budget_bytes=budget_bytes,
        reserve_frac=reserve_frac,
        logger=logger,
    )

    if n_jobs != n_jobs_req:
        logger.info("Using n_jobs=%d after clamping.", n_jobs)
    else:
        logger.info("Using n_jobs=%d (no clamping needed).", n_jobs)

    return n_jobs