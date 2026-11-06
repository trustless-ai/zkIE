#!/usr/bin/env python3
"""O5 measurement harness: cold/warm page-cache metering.

Usage: o5_measure.py <binary> <out.log> [cache_probe_file ...]

- Probes page-cache residency with `fincore --bytes` for every probe file
  BEFORE spawning the child (so a post-drop_caches run can prove the cache was
  cold) and AFTER it exits.
- Samples /proc/<pid>/io (read_bytes/write_bytes), /proc/<pid>/status
  (VmHWM/VmRSS) and /proc/<pid>/stat majflt while the child runs.
- Reports child rusage (maxrss/minflt/majflt/inblock/oublock) at exit.

Cold-read accounting on Linux: read_bytes and ru_inblock count read(2)/write(2)
I/O; memory-mapped page-in I/O is charged to major faults instead of
read_bytes on kernels that do not account filemap I/O to the faulting task.
The RESULT line therefore reports majflt together with read_bytes/inblock, and
fincore_before/fincore_after state the page-cache residency at run time. A cold
run is only claimed when fincore_before is 0 for the probed file and the
corresponding I/O counter (read_bytes for read(2) paths, majflt for mmap paths)
is non-zero.
"""
import os
import resource
import subprocess
import sys
import time


def fincore(path):
    try:
        r = subprocess.run(
            ["fincore", "--bytes", path], capture_output=True, text=True, check=False
        )
        line = r.stdout.strip().splitlines()[-1].split()
        return int(line[0])
    except Exception:
        return -1


def main():
    binary, out = sys.argv[1], sys.argv[2]
    probes = sys.argv[3:]
    fc_before = [fincore(p) for p in probes]

    log = open(out, "w")
    log.write("# cmd: {}\n".format(" ".join([binary] + probes)))
    log.write(
        "# env: DS_WEIGHTS={} ZKIE_CACHE={} DS_LAYER={}\n".format(
            os.environ.get("DS_WEIGHTS", ""),
            os.environ.get("ZKIE_CACHE", ""),
            os.environ.get("DS_LAYER", ""),
        )
    )
    log.write("# fincore_before_bytes: {}\n".format(fc_before))
    log.flush()

    t0 = time.time()
    p = subprocess.Popen([binary], stdout=log, stderr=subprocess.STDOUT)
    last = {}
    peak_hwm = peak_rss = peak_maj = 0
    while p.poll() is None:
        try:
            with open("/proc/{}/io".format(p.pid)) as f:
                for line in f:
                    k, v = line.split(":")
                    last[k.strip()] = int(v)
            with open("/proc/{}/status".format(p.pid)) as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        peak_hwm = max(peak_hwm, int(line.split()[1]))
                    elif line.startswith("VmRSS:"):
                        peak_rss = max(peak_rss, int(line.split()[1]))
            with open("/proc/{}/stat".format(p.pid)) as f:
                st = f.read()
            rest = st[st.rfind(")") + 2:].split()
            if len(rest) > 9:
                peak_maj = max(peak_maj, int(rest[9]))
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            pass
        time.sleep(0.05)
    rc = p.wait()
    wall = time.time() - t0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    fc_after = [fincore(p) for p in probes]
    line = (
        "RESULT rc={} wall_s={:.3f} maxrss_kb={} minflt={} majflt={} "
        "sampled_majflt_max={} inblock={} oublock={} sampled_read_bytes={} "
        "sampled_write_bytes={} peak_sampled_hwm_kb={} peak_sampled_rss_kb={} "
        "fincore_before={} fincore_after={}"
    ).format(
        rc,
        wall,
        ru.ru_maxrss,
        ru.ru_minflt,
        ru.ru_majflt,
        peak_maj,
        ru.ru_inblock,
        ru.ru_oublock,
        last.get("read_bytes", -1),
        last.get("write_bytes", -1),
        peak_hwm,
        peak_rss,
        ",".join(map(str, fc_before)) if fc_before else "-",
        ",".join(map(str, fc_after)) if fc_after else "-",
    )
    print(line)
    log.write(line + "\n")
    log.close()


if __name__ == "__main__":
    main()
