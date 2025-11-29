"""
Cleaned Google Custom Search image helper.

Ready for deployment / push to GitHub. Requires GOOGLE_API_KEY and GOOGLE_CX
in the environment (use a .env file or environment variables).
"""

import os
import time
import json
import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()  # ensure environment variables from .env are loaded

# Config / env
API_KEY = os.getenv("GOOGLE_API_KEY")
CX = os.getenv("GOOGLE_CX")

# default HTTP headers
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; image-downloader/1.0)"}

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def require_env() -> None:
    """Raise a RuntimeError when required environment variables are missing."""
    missing = [k for k, v in {"GOOGLE_API_KEY": API_KEY, "GOOGLE_CX": CX}.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}. Did you call load_dotenv()?")


def api_preview_images(
    query: str,
    num: int = 20,
    gl: str = "ca",
    hl: str = "en",
    lr: str = "lang_en",
    safe: str = "off",
    imgType: str = "photo",
    rights: Optional[str] = None,
    imgSize: Optional[str] = None,
) -> None:
    """
    Fetch up to `num` image results from Google Custom Search and show them
    using matplotlib. This function imports matplotlib lazily so the module
    can be used on servers without display if preview functionality is not used.
    """
    require_env()

    try:
        import matplotlib.pyplot as plt  # local import to avoid headless import errors
    except Exception as e:
        raise RuntimeError("matplotlib is required for previewing images") from e

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CX,
        "q": query,
        "searchType": "image",
        "num": min(num, 10),
        "gl": gl,
        "hl": hl,
        "lr": lr,
        "safe": safe,
        "imgType": imgType,
    }
    if rights:
        params["rights"] = rights
    if imgSize:
        params["imgSize"] = imgSize

    items = []
    start = 1
    while len(items) < num:
        params["start"] = start
        r = requests.get(url, params=params, timeout=20, headers=DEFAULT_HEADERS)
        r.raise_for_status()
        data = r.json()
        new_items = data.get("items", [])
        if not new_items:
            break
        items.extend(new_items)
        start += 10
        if start > 91:
            break

    logger.info("Retrieved %d image results for '%s'", len(items), query)

    cols = 5
    rows = (num + cols - 1) // cols
    plt.figure(figsize=(15, rows * 3))
    for i, it in enumerate(items[:num]):
        link = it.get("link")
        if not link:
            continue
        try:
            img_data = requests.get(link, timeout=10, headers=DEFAULT_HEADERS).content
            img = Image.open(BytesIO(img_data)).convert("RGB")
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{i + 1}", fontsize=8)
        except Exception:
            continue
    plt.suptitle(f"Google PSE preview: '{query}'", fontsize=14)
    plt.tight_layout()
    plt.show()


def google_image_search(
    query: str,
    *,
    start: int = 1,
    num: int = 10,
    rights: Optional[str] = None,
    imgSize: Optional[str] = None,
    imgType: str = "photo",
    gl: str = "ca",
    hl: str = "en",
    lr: str = "lang_en",
    safe: str = "off",
    exactTerms: Optional[str] = None,
    siteSearch: Optional[str] = None,
) -> List[Dict]:
    """Return up to `num` image results (max 10 per API call)."""
    require_env()
    params = {
        "key": API_KEY,
        "cx": CX,
        "q": query,
        "searchType": "image",
        "num": min(num, 10),
        "start": start,
        "gl": gl,
        "hl": hl,
        "lr": lr,
        "safe": safe,
        "imgType": imgType,
    }
    if rights:
        params["rights"] = rights
    if imgSize:
        params["imgSize"] = imgSize
    if exactTerms:
        params["exactTerms"] = exactTerms
    if siteSearch:
        params["siteSearch"] = siteSearch

    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=20, headers=DEFAULT_HEADERS)
    if r.status_code != 200:
        try:
            logger.error("API error payload: %s", r.json())
        except Exception:
            logger.error("API error text: %s", r.text)
        r.raise_for_status()
    return r.json().get("items", [])


def _save_image(content: bytes, out_dir: Path, base_name: str) -> Tuple[Optional[Path], Optional[str], Optional[Tuple[int, int]]]:
    """Save image content to out_dir with a hashed filename. Returns (path, md5, (w,h))."""
    md5 = hashlib.md5(content).hexdigest()
    fn = f"{base_name}_{md5[:10]}.jpg"
    path = out_dir / fn
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        return None, None, None

    # filter out very small images (often icons)
    if min(img.size) < 128:
        return None, None, None

    img.save(path, "JPEG", quality=90)
    return path, md5, img.size


def download_images(
    query: str,
    out_dir: str,
    total: int = 50,
    *,
    rights: Optional[str] = "cc_publicdomain|cc_attribute|cc_sharealike",
    imgSize: Optional[str] = None,
    imgType: str = "photo",
    gl: str = "ca",
    hl: str = "en",
    lr: str = "lang_en",
    safe: str = "off",
    exactTerms: Optional[str] = None,
    siteSearch: Optional[str] = None,
    delay: float = 0.2,
) -> None:
    """
    Download images for `query` into out_dir and write metadata.jsonl.
    Avoids duplicates by URL and MD5.
    """
    require_env()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    meta_fp_path = out_path / "metadata.jsonl"

    start = 1
    grabbed = 0
    seen_urls = set()
    seen_md5 = set()

    with meta_fp_path.open("a", encoding="utf-8") as meta_fp:
        while grabbed < total and start <= 91:
            remaining = total - grabbed
            items = google_image_search(
                query,
                start=start,
                num=min(10, remaining),
                rights=rights,
                imgSize=imgSize,
                imgType=imgType,
                gl=gl,
                hl=hl,
                lr=lr,
                safe=safe,
                exactTerms=exactTerms,
                siteSearch=siteSearch,
            )
            if not items:
                break

            for it in items:
                if grabbed >= total:
                    break
                url = it.get("link")
                if not url or url in seen_urls:
                    continue

                for attempt in range(2):
                    try:
                        resp = requests.get(url, timeout=15, headers=DEFAULT_HEADERS)
                        resp.raise_for_status()
                        content_type = resp.headers.get("Content-Type", "")
                        if "image" not in content_type:
                            raise ValueError("Not an image Content-Type")
                        path, md5, size = _save_image(resp.content, out_path, base_name=query.replace(" ", "_"))
                        if not path:
                            raise ValueError("Image too small or unreadable")
                        if md5 in seen_md5:
                            # duplicate content; remove file we just wrote if exists
                            if path.exists():
                                path.unlink(missing_ok=True)
                            raise ValueError("Duplicate by MD5")
                        seen_md5.add(md5)
                        seen_urls.add(url)

                        meta = {
                            "file": path.name,
                            "query": query,
                            "url": url,
                            "displayLink": it.get("displayLink"),
                            "contextLink": (it.get("image") or {}).get("contextLink"),
                            "mime": it.get("mime"),
                            "width": size[0],
                            "height": size[1],
                            "md5": md5,
                            "params": {
                                "rights": rights,
                                "imgSize": imgSize,
                                "imgType": imgType,
                                "gl": gl,
                                "hl": hl,
                                "lr": lr,
                                "safe": safe,
                                "exactTerms": exactTerms,
                                "siteSearch": siteSearch,
                            },
                        }
                        meta_fp.write(json.dumps(meta) + "\n")
                        meta_fp.flush()
                        grabbed += 1
                        logger.info("[%d/%d] saved %s (%dx%d)", grabbed, total, meta["file"], size[0], size[1])
                        break
                    except Exception:
                        if attempt == 1:
                            # final failure — skip this URL
                            logger.debug("Skipping URL after retries: %s", url)
                        time.sleep(0.1)

            start += 10
            time.sleep(delay)

    logger.info("%s: saved %d images to %s", query, grabbed, out_path)


def fn(fruit: str, adjs: Optional[List[str]] = None, fresh: bool = True) -> None:
    """Helper to bulk download example datasets for a `fruit`."""
    if adjs is None:
        adjs = ["rotten", "damaged", "moldy", "bruised", "overripe"]

    if fresh:
        download_images(fruit, out_dir=f"dataset/{fruit}_good", total=50, rights=None, imgSize=None, imgType="photo")
        download_images(f"fresh {fruit}", out_dir=f"dataset/{fruit}_good", total=100, rights=None, imgSize=None, imgType="photo")

    for j in adjs:
        download_images(f"{j} {fruit}", out_dir=f"dataset/{fruit}_bad", total=30, rights=None, imgSize=None, imgType="photo")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Google Custom Search image helper")
    sub = parser.add_subparsers(dest="cmd")

    p_preview = sub.add_parser("preview", help="Preview images for a query (opens a matplotlib window)")
    p_preview.add_argument("query", help="Search query")
    p_preview.add_argument("--num", type=int, default=20, help="Number of images to preview")

    p_download = sub.add_parser("download", help="Download images for a query")
    p_download.add_argument("query", help="Search query")
    p_download.add_argument("out_dir", help="Directory to save images")
    p_download.add_argument("--total", type=int, default=50, help="Total images to download")

    args = parser.parse_args()
    if args.cmd == "preview":
        api_preview_images(args.query, num=args.num)
    elif args.cmd == "download":
        download_images(args.query, args.out_dir, total=args.total)
    else:
        parser.print_help()