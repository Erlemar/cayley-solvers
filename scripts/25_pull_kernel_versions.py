"""Download a named output file from EVERY available version of a Kaggle kernel.

Uses ApiDownloadKernelOutputRequest.version_number, which (unlike `kaggle kernels
output`) is version-addressable. A 404 means that version produced no such file
(failed / still running / different filename), not that the version is absent.
"""
import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request

from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest


def fetch(token, owner, slug, file_path, version):
    client = KaggleClient(api_token=token)
    with client as c:
        req = ApiDownloadKernelOutputRequest()
        req.owner_slug = owner
        req.kernel_slug = slug
        req.file_path = file_path
        if version is not None:
            req.version_number = version
        redirect = c.kernels.kernels_api_client.download_kernel_output(req)
    url = getattr(redirect, "url", None) or str(redirect)
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, help="owner/slug")
    ap.add_argument("--file", required=True, help="output filename to pull")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-version", type=int, default=1)
    ap.add_argument("--max-version", type=int, default=40)
    args = ap.parse_args()

    owner, slug = args.kernel.split("/", 1)
    token = os.environ["KAGGLE_API_TOKEN"]
    os.makedirs(args.out_dir, exist_ok=True)

    ok, missing, errs = [], [], []
    seen = {}
    for v in range(args.min_version, args.max_version + 1):
        try:
            data = fetch(token, owner, slug, args.file, v)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                missing.append(v)
                print("v%-3d 404 (no such output)" % v, flush=True)
            else:
                errs.append((v, "HTTP %d" % e.code))
                print("v%-3d HTTP %d" % (v, e.code), flush=True)
            continue
        except Exception as e:  # noqa: BLE001
            errs.append((v, type(e).__name__))
            print("v%-3d ERROR %s: %s" % (v, type(e).__name__, str(e)[:120]), flush=True)
            continue

        digest = hashlib.md5(data).hexdigest()
        dest = os.path.join(args.out_dir, "v%03d_%s" % (v, args.file))
        with open(dest, "wb") as fh:
            fh.write(data)
        dup = seen.get(digest)
        seen.setdefault(digest, v)
        ok.append(v)
        print("v%-3d OK  %8d bytes  md5=%s%s"
              % (v, len(data), digest, "  (== v%d)" % dup if dup else ""),
              flush=True)

    print("\n=== summary ===")
    print("downloaded : %d versions -> %s" % (len(ok), sorted(ok)))
    print("distinct   : %d unique payloads" % len(seen))
    print("404 (none) : %d versions -> %s" % (len(missing), sorted(missing)))
    if errs:
        print("errors     : %s" % errs)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
