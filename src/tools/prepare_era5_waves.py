#!/usr/bin/env python3
"""
Prepare ERA5 wave inputs for the Fortran oil-spill model.

This script intentionally does not hard-code any case-specific time range,
domain, or timestep. Those values should be supplied by a later case runner
or by explicit CLI arguments.

Typical workflow:

  1. Download ERA5 wave fields to a cached NetCDF file.
  2. Optionally convert that NetCDF file to the model's raw Fortran binary
     layout: float64 values, shape (nx, ny, nt), flattened in Fortran order.

The conversion keeps ERA5 mean wave direction in its native convention by
default. Do not switch wave-direction convention here unless it has been
confirmed against the MLP training feature code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence


DATASET = "reanalysis-era5-single-levels"

ERA5_WAVE_VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_period",
    "mean_wave_direction",
]

OUTPUT_VARIABLES = {
    "swh": (
        "swh",
        "significant_height_of_combined_wind_waves_and_swell",
    ),
    "mwp": (
        "mwp",
        "mean_wave_period",
    ),
    "wave_dir": (
        "mwd",
        "mean_wave_direction",
        "wave_dir",
    ),
}


@dataclass(frozen=True)
class Era5WaveConfig:
    start: datetime
    end: datetime
    area: tuple[float, float, float, float]
    hours: tuple[str, ...] | None
    data_format: str = "netcdf"
    download_format: str = "unarchived"


@dataclass(frozen=True)
class TargetGrid:
    longitude: object
    latitude: object
    offsets_hours: object
    target_times: object


def parse_datetime(value: str) -> datetime:
    """Parse a compact ISO-like timestamp accepted by the CLI."""
    normalized = value.strip().replace(" ", "T")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"invalid datetime '{value}', expected YYYY-MM-DD or YYYY-MM-DDTHH[:MM[:SS]]"
    )


def normalize_hour(value: str) -> str:
    """Return CDS-style HH:MM hour strings."""
    text = value.strip()
    if ":" in text:
        hour_text, minute_text = text.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    else:
        hour = int(text)
        minute = 0
    if not 0 <= hour <= 23 or minute != 0:
        raise argparse.ArgumentTypeError(
            f"invalid hour '{value}', expected whole-hour values from 0 to 23"
        )
    return f"{hour:02d}:00"


def parse_hours(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    expanded: list[str] = []
    for value in values:
        expanded.extend(part for part in value.split(",") if part.strip())
    return tuple(sorted({normalize_hour(value) for value in expanded}))


def iter_dates(start: datetime, end: datetime) -> Iterable[datetime]:
    current = datetime(start.year, start.month, start.day)
    last = datetime(end.year, end.month, end.day)
    while current <= last:
        yield current
        current += timedelta(days=1)


def iter_hourly_times(start: datetime, end: datetime) -> Iterable[datetime]:
    current = start
    while current <= end:
        yield current
        current += timedelta(hours=1)


def build_request(config: Era5WaveConfig) -> dict[str, object]:
    """Build the CDS API request for ERA5 wave variables."""
    if config.end < config.start:
        raise ValueError("--end must be greater than or equal to --start")

    dates = list(iter_dates(config.start, config.end))
    if config.hours is None:
        hours = tuple(sorted({timestamp.strftime("%H:00") for timestamp in iter_hourly_times(config.start, config.end)}))
    else:
        hours = config.hours

    return {
        "product_type": ["reanalysis"],
        "variable": ERA5_WAVE_VARIABLES,
        "year": sorted({f"{date.year:04d}" for date in dates}),
        "month": sorted({f"{date.month:02d}" for date in dates}),
        "day": sorted({f"{date.day:02d}" for date in dates}),
        "time": list(hours),
        "area": list(config.area),
        "data_format": config.data_format,
        "download_format": config.download_format,
    }


def cache_filename(config: Era5WaveConfig) -> str:
    north, west, south, east = config.area
    area_slug = f"N{north:g}_W{west:g}_S{south:g}_E{east:g}".replace(".", "p").replace("-", "m")
    suffix = "nc" if config.data_format == "netcdf" else config.data_format
    return (
        f"era5_waves_{config.start:%Y%m%d%H}_"
        f"{config.end:%Y%m%d%H}_{area_slug}.{suffix}"
    )


def download_era5_waves(request: dict[str, object], cache_path: Path, overwrite: bool = False) -> Path:
    """Download ERA5 waves with cdsapi, using cache_path as the target file."""
    if cache_path.exists() and not overwrite:
        print(f"[skip] cache exists: {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is required for download. Install in WSL/conda with: "
            "pip install \"cdsapi>=0.7.7\""
        ) from exc

    print(f"[download] dataset={DATASET}")
    print(f"[download] target={cache_path}")
    client = cdsapi.Client()
    client.retrieve(DATASET, request, str(cache_path))
    return cache_path


def load_target_grid(input_dir: Path, file_id: str, reference_time: datetime) -> TargetGrid:
    """Read model longitude, latitude, and relative time files."""
    import numpy as np

    lon_path = input_dir / "longitude.dat"
    lat_path = input_dir / "latitude.dat"
    time_path = input_dir / f"time_{file_id}.dat"

    missing = [str(path) for path in (lon_path, lat_path, time_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing model input file(s): " + ", ".join(missing))

    longitude = np.loadtxt(lon_path, dtype=np.float64)
    latitude = np.loadtxt(lat_path, dtype=np.float64)
    offsets_hours = np.loadtxt(time_path, dtype=np.float64)
    offsets_hours = np.atleast_1d(offsets_hours)
    target_times = np.array(
        [reference_time + timedelta(hours=float(offset)) for offset in offsets_hours],
        dtype="datetime64[ns]",
    )

    return TargetGrid(
        longitude=longitude,
        latitude=latitude,
        offsets_hours=offsets_hours,
        target_times=target_times,
    )


def find_coord_name(dataset: object, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name
    available = sorted(set(dataset.coords) | set(dataset.dims))
    raise KeyError(f"could not find coordinate from {candidates}; available={available}")


def find_data_var(dataset: object, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in dataset.data_vars:
            return name
    available = sorted(dataset.data_vars)
    raise KeyError(f"could not find data variable from {candidates}; available={available}")


def convert_to_fortran_dat(
    nc_path: Path,
    input_dir: Path,
    out_dir: Path,
    file_id: str,
    reference_time: datetime,
    overwrite: bool = False,
) -> list[Path]:
    """Convert ERA5 NetCDF wave variables to raw Fortran binary files."""
    import numpy as np
    import xarray as xr

    grid = load_target_grid(input_dir, file_id, reference_time)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    lon_name = find_coord_name(ds, ("longitude", "lon"))
    lat_name = find_coord_name(ds, ("latitude", "lat"))
    time_name = find_coord_name(ds, ("valid_time", "time"))

    # xarray interpolation expects monotonic coordinates. ERA5 latitude is
    # often descending, while the model input files here are ascending.
    if ds[lon_name].size > 1 and bool((ds[lon_name].diff(lon_name) < 0).any().item()):
        ds = ds.sortby(lon_name)
    if ds[lat_name].size > 1 and bool((ds[lat_name].diff(lat_name) < 0).any().item()):
        ds = ds.sortby(lat_name)

    outputs: list[Path] = []
    expected_size = int(len(grid.longitude) * len(grid.latitude) * len(grid.offsets_hours) * 8)

    for output_name, candidates in OUTPUT_VARIABLES.items():
        var_name = find_data_var(ds, candidates)
        target_path = out_dir / f"{output_name}_{file_id}.dat"
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"output exists, pass --overwrite to replace: {target_path}")

        da = ds[var_name].interp(
            {
                lon_name: grid.longitude,
                lat_name: grid.latitude,
                time_name: grid.target_times,
            }
        )
        da = da.transpose(lon_name, lat_name, time_name)
        values = np.asarray(da.values, dtype=np.float64)
        values = np.where(np.isfinite(values), values, 9999.0)

        if values.shape != (len(grid.longitude), len(grid.latitude), len(grid.offsets_hours)):
            raise ValueError(
                f"{output_name} has unexpected shape {values.shape}; "
                f"expected {(len(grid.longitude), len(grid.latitude), len(grid.offsets_hours))}"
            )

        values.ravel(order="F").tofile(target_path)
        actual_size = target_path.stat().st_size
        if actual_size != expected_size:
            raise IOError(
                f"{target_path} size mismatch: got {actual_size}, expected {expected_size}"
            )

        print(
            f"[write] {target_path} shape={values.shape} "
            f"dtype=float64 min={float(np.nanmin(values)):.6g} max={float(np.nanmax(values)):.6g}"
        )
        outputs.append(target_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERA5 wave fields and optionally convert them to Fortran raw .dat files."
    )
    parser.add_argument("--start", type=parse_datetime, help="case/reference start time, e.g. YYYY-MM-DDTHH")
    parser.add_argument("--end", type=parse_datetime, help="download end time, e.g. YYYY-MM-DDTHH")
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="CDS area box in north west south east order",
    )
    parser.add_argument(
        "--hours",
        nargs="+",
        help="optional whole-hour list, e.g. --hours 00 01 02 or --hours 00,01,02",
    )
    parser.add_argument("--out-dir", type=Path, help="directory for generated swh/mwp/wave_dir .dat files")
    parser.add_argument("--input-dir", type=Path, help="directory containing longitude/latitude/time files")
    parser.add_argument("--file-id", default="0001", help="input/output file id, default: 0001")
    parser.add_argument("--cache-dir", type=Path, default=Path("era5_cache"), help="NetCDF cache directory")
    parser.add_argument("--nc-path", type=Path, help="existing ERA5 NetCDF file; skips download")
    parser.add_argument("--download-only", action="store_true", help="download NetCDF only, do not convert")
    parser.add_argument("--convert", action="store_true", help="convert NetCDF to Fortran .dat files")
    parser.add_argument("--overwrite", action="store_true", help="replace existing cache/output files")
    parser.add_argument("--dry-run", action="store_true", help="print CDS request and exit without downloading")
    return parser.parse_args()


def require_download_args(args: argparse.Namespace) -> Era5WaveConfig:
    missing = []
    if args.start is None:
        missing.append("--start")
    if args.end is None:
        missing.append("--end")
    if args.area is None:
        missing.append("--area")
    if missing:
        raise SystemExit("download requires: " + ", ".join(missing))
    return Era5WaveConfig(
        start=args.start,
        end=args.end,
        area=tuple(args.area),
        hours=parse_hours(args.hours),
    )


def main() -> int:
    args = parse_args()
    if args.download_only and args.convert:
        raise SystemExit("--download-only and --convert are mutually exclusive")

    nc_path = args.nc_path
    request = None

    if nc_path is None:
        config = require_download_args(args)
        request = build_request(config)
        nc_path = args.cache_dir / cache_filename(config)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": DATASET,
                    "request": request,
                    "target": str(nc_path),
                    "will_convert": bool(args.convert),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.nc_path is None:
        assert request is not None
        nc_path = download_era5_waves(request, nc_path, overwrite=args.overwrite)

    if args.download_only or not args.convert:
        print(f"[done] NetCDF ready: {nc_path}")
        return 0

    if args.start is None:
        raise SystemExit("--convert requires --start as the reference time for time_XXXX.dat offsets")
    if args.out_dir is None:
        raise SystemExit("--convert requires --out-dir")

    input_dir = args.input_dir or args.out_dir
    convert_to_fortran_dat(
        nc_path=nc_path,
        input_dir=input_dir,
        out_dir=args.out_dir,
        file_id=args.file_id,
        reference_time=args.start,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
