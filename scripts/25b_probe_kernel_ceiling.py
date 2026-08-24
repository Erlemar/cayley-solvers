"""Find the highest existing version of a Kaggle kernel in ~10 requests.

Companion to `25_pull_kernel_versions.py`: probe the ceiling first, then pull
`--min-version <last_pulled+1> --max-version <ceiling>` instead of scanning blind.

Mechanism (CLAUDE.md 7f(d)) -- the two 404 sources mean different things:
  404 from www.kaggleusercontent.com          -> version EXISTS, no such output file
  404 from api.kaggle.com/.../DownloadKernelOutput -> no such version

Usage:
    python scripts/25b_probe_kernel_ceiling.py "owner/slug::submission.csv" ...
"""
import os
import sys
import urllib.error
import urllib.request

from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

TOKEN = os.environ["KAGGLE_API_TOKEN"]


def probe(owner, slug, file_path, version):
    """Return 'ok' | 'exists' | 'absent' | 'err:<type>'."""
    try:
        with KaggleClient(api_token=TOKEN) as c:
            req = ApiDownloadKernelOutputRequest()
            req.owner_slug, req.kernel_slug = owner, slug
            req.file_path, req.version_number = file_path, version
            redirect = c.kernels.kernels_api_client.download_kernel_output(req)
    except urllib.error.HTTPError as e:
        return "absent" if "api.kaggle.com" in str(e.url or "") else "exists"
    except Exception as e:                                  # noqa: BLE001
        return "absent" if "404" in str(e) else "err:%s" % type(e).__name__
    url = getattr(redirect, "url", None) or str(redirect)
    try:
        with urllib.request.urlopen(url) as resp:
            resp.read(64)
        return "ok"
    except urllib.error.HTTPError as e:
        return "exists" if "kaggleusercontent" in (e.url or "") else "absent"


def ceiling(owner, slug, file_path, lo=1, hi=200):
    """Highest version that is not 'absent'; 0 if the kernel has none."""
    if probe(owner, slug, file_path, lo) == "absent":
        return 0
    while probe(owner, slug, file_path, hi) != "absent":
        lo, hi = hi, hi * 2
        if hi > 4096:
            break
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if probe(owner, slug, file_path, mid) == "absent":
            hi = mid
        else:
            lo = mid
    return lo


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for spec in sys.argv[1:]:
        kernel, _, fp = spec.partition("::")
        fp = fp or "submission.csv"
        owner, slug = kernel.split("/", 1)
        print("%-55s %-28s max_version=%d"
              % (kernel, fp, ceiling(owner, slug, fp)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
