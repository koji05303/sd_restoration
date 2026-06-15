from __future__ import annotations

import argparse
import gc
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CUDA_DEVICE = "cuda:0"  
DEFAULT_OUTPUT_DIR = Path("output")
VALID_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

"""
TODO:

"""


@dataclass
class BatchResult:
    input_path: Path
    output_path: Optional[Path]
    success: bool
    error: str = ""
    elapsed_seconds: float = 0.0
    input_size: Optional[tuple[int, int]] = None
    output_size: Optional[tuple[int, int]] = None


BatchProgressCallback = Callable[[int, int, Path, Optional[Path], int, int], None]


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def collect_input_images(
    image_paths: Iterable[Path] = (),
    input_dir: Optional[Path] = None,
    recursive: bool = False,
) -> list[Path]:
    images: list[Path] = []

    for image_path in image_paths:
        if not is_supported_image(image_path):
            raise ValueError(f"Unsupported or missing image file: {image_path}")
        images.append(image_path)

    if input_dir is not None:
        if not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
        images.extend(path for path in iterator if is_supported_image(path))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for image_path in sorted(images, key=lambda item: str(item).lower()):
        resolved = image_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(image_path)

    return deduped


def release_device_memory(device: torch.device | str = DEFAULT_CUDA_DEVICE) -> None:
    gc.collect()
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError):
        torch_device = torch.device("cpu")

    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom_error(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def reduced_tile_size(tile_size: int, min_tile_size: int = 64) -> Optional[int]:
    if tile_size <= min_tile_size:
        return None
    next_tile_size = max(min_tile_size, (tile_size // 2) // 8 * 8)
    if next_tile_size >= tile_size:
        return None
    return next_tile_size


def read_bgr_image(image_path: Path) -> np.ndarray:
    buffer = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def write_bgr_image(image_path: Path, image_bgr: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower()
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 96]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    ok, encoded = cv2.imencode(suffix, image_bgr, params)
    if not ok:
        raise ValueError(f"Could not encode output image as {suffix}: {image_path}")
    encoded.tofile(str(image_path))


def output_suffix_for(input_path: Path, output_format: str) -> str:
    normalized_format = (output_format or "png").strip().lower()
    if normalized_format == "jpg":
        return ".jpg"
    if normalized_format == "keep":
        suffix = input_path.suffix.lower()
        return suffix if suffix in VALID_IMAGE_EXTENSIONS else ".png"
    return ".png"


def resolve_batch_output_path(
    input_path: Path,
    output_dir: Path,
    output_format: str = "png",
    input_root: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    relative_parent = Path()
    if input_root is not None:
        try:
            relative_parent = input_path.resolve().parent.relative_to(input_root.resolve())
        except ValueError:
            relative_parent = Path()

    output_path = (
        output_dir
        / relative_parent
        / f"{input_path.stem}_pure_psr{output_suffix_for(input_path, output_format)}"
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    return output_path


def enhance_path(
    engine: "PureSREngine",
    input_path: Path,
    output_path: Path,
    enhance_detail: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    retry_on_oom: bool = True,
    min_tile_size: int = 64,
) -> tuple[tuple[int, int], tuple[int, int]]:
    while True:
        input_bgr: Optional[np.ndarray] = None
        output_bgr: Optional[np.ndarray] = None
        try:
            input_bgr = read_bgr_image(input_path)
            height, width = input_bgr.shape[:2]
            output_bgr = engine.enhance(
                input_bgr,
                enhance_detail=enhance_detail,
                progress_callback=progress_callback,
            )
            out_height, out_width = output_bgr.shape[:2]
            write_bgr_image(output_path, output_bgr)
            return (width, height), (out_width, out_height)
        except RuntimeError as exc:
            next_tile_size = reduced_tile_size(int(engine.tile_size), min_tile_size)
            if not retry_on_oom or not is_cuda_oom_error(exc) or next_tile_size is None:
                raise
            print(
                f"[Warning] CUDA OOM at tile_size={engine.tile_size}. "
                f"Retrying {input_path} with tile_size={next_tile_size}."
            )
            engine.tile_size = next_tile_size
            continue
        finally:
            del input_bgr, output_bgr
            release_device_memory(engine.device)


def run_batch(
    engine: "PureSREngine",
    image_paths: Iterable[Path],
    output_dir: Path,
    output_format: str = "png",
    enhance_detail: bool = True,
    overwrite: bool = False,
    input_root: Optional[Path] = None,
    progress_callback: Optional[BatchProgressCallback] = None,
) -> list[BatchResult]:
    paths = list(image_paths)
    results: list[BatchResult] = []
    total_images = len(paths)

    for image_index, input_path in enumerate(paths, start=1):
        output_path: Optional[Path] = None
        start_time = time.monotonic()
        try:
            output_path = resolve_batch_output_path(
                input_path=input_path,
                output_dir=output_dir,
                output_format=output_format,
                input_root=input_root,
                overwrite=overwrite,
            )
            if progress_callback:
                progress_callback(image_index, total_images, input_path, output_path, 0, 0)

            def update_tile_progress(done: int, total: int) -> None:
                if progress_callback:
                    progress_callback(image_index, total_images, input_path, output_path, done, total)

            input_size, output_size = enhance_path(
                engine=engine,
                input_path=input_path,
                output_path=output_path,
                enhance_detail=enhance_detail,
                progress_callback=update_tile_progress,
            )
            results.append(
                BatchResult(
                    input_path=input_path,
                    output_path=output_path,
                    success=True,
                    elapsed_seconds=time.monotonic() - start_time,
                    input_size=input_size,
                    output_size=output_size,
                )
            )
        except Exception as exc:
            release_device_memory(engine.device)
            results.append(
                BatchResult(
                    input_path=input_path,
                    output_path=output_path,
                    success=False,
                    error=str(exc),
                    elapsed_seconds=time.monotonic() - start_time,
                )
            )

    return results


# ==========================================
# 手刻神經網路架構 (RRDBNet)
# 對齊 Real-ESRGAN_x4plus 的官方權重矩陣
# ==========================================

class DenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x # 殘差縮放

class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = DenseBlock(num_feat, num_grow_ch)
        self.rdb2 = DenseBlock(num_feat, num_grow_ch)
        self.rdb3 = DenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class CustomRRDBNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_feat, num_grow_ch, num_block = 64, 32, 23
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 兩次雙倍上採樣, total x4
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # 上採樣
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out



### PSR Engine, code from scratch
class PureSREngine:
    def __init__(self, device=DEFAULT_CUDA_DEVICE, model_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"):
        self.device = torch.device(device)
        self.scale = 4
        # fp16, 省省省屁眼汁
        self.half = self.device.type == "cuda" and torch.cuda.is_available()
                
        # 由於開啟了 half 精度，tile_size可以高一咪咪, 建議是別開太大，我先抓256 避免屁眼開花
        self.tile_size = 256
        # 增加 padding 避免邊界偽影 and 無縫拼接
        self.tile_pad = 16   
        
        print(f"Pure SR 引擎 ==> 綁定設備: {device} | 精度: {'FP16' if self.half else 'FP32'}")
        self.model = CustomRRDBNet()
        self._load_weights(model_url) 

        self.model.eval()
        self.model.requires_grad_(False)

        if self.half:
            self.model = self.model.half()

        self.model = self.model.to(self.device)

    def _load_weights(self, url):
        
        weight_path = "RealESRGAN_x4plus.pth"

        if not os.path.exists(weight_path):
            print(f"Pulling Original weights from {url}...")
            urllib.request.urlretrieve(url, weight_path)
            
        print("Recreating the model arc, Matrix loading...")
        
        loadnet = torch.load(weight_path, map_location="cpu")
        
        keyname = "params_ema" if "params_ema" in loadnet else "params"
        self.model.load_state_dict(loadnet[keyname], strict=True)

        del loadnet
        gc.collect()
        
        print("Node Inject complete.")

    @torch.inference_mode()
    def enhance(self, img_bgr, enhance_detail=True, progress_callback=None):
        """輸入 OpenCV BGR 圖片，吐出 4 倍無損放大圖片"""
        #BGR -> RGB -> Float Tensor [1, 3, H, W]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        if self.half:
            img_tensor = img_tensor.half()

        _, channel, height, width = img_tensor.shape
        output_height = height * self.scale
        output_width = width * self.scale
        
        ## Canvas stays in CPU memory as uint8 to reduce host RAM pressure on large outputs.
        output_numpy_rgb = np.empty((output_height, output_width, channel), dtype=np.uint8)
        total_tiles = ((height + self.tile_size - 1) // self.tile_size) * (
            (width + self.tile_size - 1) // self.tile_size
        )
        tile_idx = 0
        cuda_clean_interval = 8
        gc_clean_interval = 32

        # sliding window with tile and padding to process large images without OOM,
        for y in range(0, height, self.tile_size):
            for x in range(0, width, self.tile_size):
                tile_idx += 1
                # 取得當前區塊的邊界
                y_end = min(y + self.tile_size, height)
                x_end = min(x + self.tile_size, width)
                
                # 計算包含 padding 的擴展邊界 (處理邊緣不越界)
                y_pad_start = max(y - self.tile_pad, 0)
                y_pad_end = min(y_end + self.tile_pad, height)
                x_pad_start = max(x - self.tile_pad, 0)
                x_pad_end = min(x_end + self.tile_pad, width)
                
                # 切出帶有 padding 的小張量送去推理, stays on CPU
                input_tile = img_tensor[:, :, y_pad_start:y_pad_end, x_pad_start:x_pad_end]      

                ### send tile to GPU for processing
                input_tile = input_tile.to(self.device)
                if self.half:
                    input_tile = input_tile.half()

              
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.half):  # 丟進自建的 RRDBNet 進行前向爆破 (搭配 AMP 或純 FP16)
                    output_tile = self.model(input_tile)
                    
                # 算完了是吧 那把剛padding的部份切掉 只留下原圖區域
                y_out_start = y * self.scale
                y_out_end = y_end * self.scale
                x_out_start = x * self.scale
                x_out_end = x_end * self.scale
                
                y_crop_start = (y - y_pad_start) * self.scale
                y_crop_end = y_crop_start + (y_end - y) * self.scale
                x_crop_start = (x - x_pad_start) * self.scale
                x_crop_end = x_crop_start + (x_end - x) * self.scale
                
                # Crop on GPU first, then move only the valid area back to CPU.
                output_crop = output_tile[:, :, y_crop_start:y_crop_end, x_crop_start:x_crop_end]
                output_crop = output_crop.detach().float().cpu()
                output_crop = output_crop.squeeze(0).permute(1, 2, 0).clamp_(0, 1)
                output_crop_np = (output_crop.numpy() * 255.0).round().astype(np.uint8)

                output_numpy_rgb[y_out_start:y_out_end, x_out_start:x_out_end, :] = output_crop_np

                del input_tile, output_tile, output_crop, output_crop_np
                if self.device.type == "cuda" and tile_idx % cuda_clean_interval == 0:
                    torch.cuda.empty_cache()
                if tile_idx % gc_clean_interval == 0:
                    gc.collect()
                if progress_callback:
                    progress_callback(tile_idx, total_tiles)

        # Numpy RGB -> BGR
        output_bgr = cv2.cvtColor(output_numpy_rgb, cv2.COLOR_RGB2BGR)
        
        # Unsharp Masking (USM)
        if enhance_detail:
            # Gaussaingn Blur, sigma=2.0, kernel size auto
            blur = cv2.GaussianBlur(output_bgr, (0, 0), 2.0)
            # add weighted: output + (output - blur) * 1.5, alpha=1.5, beta=-0.5
            output_bgr = cv2.addWeighted(output_bgr, 1.5, blur, -0.5, 0)
        
        return output_bgr

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pure PSR on one image or a sequential batch.")
    parser.add_argument("inputs", nargs="*", type=Path, help="Input image file(s).")
    parser.add_argument("--input-dir", type=Path, help="Directory of images for batch mode.")
    parser.add_argument("--recursive", action="store_true", help="Scan --input-dir recursively.")
    parser.add_argument("--output", type=Path, help="Output file for single-image mode.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for batch mode.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg", "keep"],
        default="png",
        help="Output format for --output-dir mode.",
    )
    parser.add_argument("--device", default=DEFAULT_CUDA_DEVICE, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size. Lower values reduce VRAM usage.")
    parser.add_argument("--tile-pad", type=int, default=16, help="Tile padding used to reduce edge artifacts.")
    parser.add_argument("--no-detail", action="store_true", help="Disable the final unsharp-mask detail pass.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    legacy_default = not args.inputs and args.input_dir is None
    if legacy_default:
        args.inputs = [Path("input/fake.jpg")]
        if args.output is None:
            args.output = Path("output/fake_pure_sr.jpg")

    if args.tile_size < 64 or args.tile_size % 8 != 0:
        print("Error: --tile-size must be at least 64 and divisible by 8.")
        return 2
    if args.tile_pad < 0 or args.tile_pad >= args.tile_size:
        print("Error: --tile-pad must be smaller than --tile-size.")
        return 2

    try:
        images = collect_input_images(args.inputs, args.input_dir, args.recursive)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    if not images:
        print("Error: no supported input images found.")
        return 2
    if args.output is not None and len(images) != 1:
        print("Error: --output can only be used with exactly one input image.")
        return 2
    if args.output is not None and args.output.exists() and not args.overwrite:
        print(f"Error: output already exists: {args.output}. Use --overwrite to replace it.")
        return 2

    print("Starting Pure SR Matrix upscaling...")
    engine = PureSREngine(device=args.device)
    engine.tile_size = int(args.tile_size)
    engine.tile_pad = int(args.tile_pad)

    try:
        if args.output is not None:
            start_time = time.monotonic()
            input_size, output_size = enhance_path(
                engine=engine,
                input_path=images[0],
                output_path=args.output,
                enhance_detail=not args.no_detail,
            )
            elapsed = time.monotonic() - start_time
            print(
                f"Generated {args.output} | "
                f"{input_size[0]}x{input_size[1]} -> {output_size[0]}x{output_size[1]} | "
                f"{elapsed:.3f}s"
            )
            return 0

        input_root = args.input_dir if args.input_dir is not None else None

        def print_progress(
            image_index: int,
            total_images: int,
            input_path: Path,
            output_path: Optional[Path],
            tile_done: int,
            tile_total: int,
        ) -> None:
            if tile_total <= 0:
                print(f"[Batch {image_index}/{total_images}] {input_path} -> {output_path}")
                return
            if tile_done == tile_total or tile_done == 1 or tile_done % 8 == 0:
                print(
                    f"[Batch {image_index}/{total_images}] "
                    f"tile {tile_done}/{tile_total}"
                )

        results = run_batch(
            engine=engine,
            image_paths=images,
            output_dir=args.output_dir,
            output_format=args.format,
            enhance_detail=not args.no_detail,
            overwrite=args.overwrite,
            input_root=input_root,
            progress_callback=print_progress,
        )
    finally:
        release_device_memory(engine.device)

    succeeded = [result for result in results if result.success]
    failed = [result for result in results if not result.success]

    print(f"Batch complete: {len(succeeded)} succeeded, {len(failed)} failed.")
    for result in succeeded:
        print(f"  OK  {result.input_path} -> {result.output_path} ({result.elapsed_seconds:.3f}s)")
    for result in failed:
        print(f"  ERR {result.input_path}: {result.error}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
