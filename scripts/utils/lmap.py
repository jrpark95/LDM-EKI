import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import io
from PIL import Image
import imageio.v2 as imageio
import os
import datetime
import re

# 기준 시각: 2011년 3월 14일 00:00:00 UTC
BASE_TIME = datetime.datetime(2011, 3, 14, 0, 0, 0)

def plot_particle_distribution(vtk_filename, 
                               region_extent=None, 
                               bins=(300, 150), 
                               use_log_scale=True):
    try:
        mesh = pv.read(vtk_filename)
    except Exception as e:
        print(f"[Error] Failed to read {vtk_filename}: {e}")
        return None

    points = mesh.points

    if points is None or points.size == 0:
        print(f"[Skip] {vtk_filename}: No point data.")
        return None

    # 필터링: 유효한 경도 범위, NaN/Inf 제거
    valid_lon_mask = (points[:, 0] >= 0) & (points[:, 0] < 180.0)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[valid_lon_mask & finite_mask]

    if points.size == 0:
        print(f"[Skip] {vtk_filename}: No valid points after filtering.")
        return None

    lons = points[:, 0]
    lats = points[:, 1]

    # 추가 방어: NaN/Inf 체크
    valid_coords = np.isfinite(lons) & np.isfinite(lats)
    lons = lons[valid_coords]
    lats = lats[valid_coords]

    if len(lons) == 0 or len(lats) == 0:
        print(f"[Skip] {vtk_filename}: No valid lat/lon values.")
        return None

    # 시각화 영역 설정
    if region_extent is None:
        lon_min, lon_max = np.min(lons), np.max(lons)
        lat_min, lat_max = np.min(lats), np.max(lats)
    else:
        lon_min, lon_max, lat_min, lat_max = region_extent

    try:
        H, lon_edges, lat_edges = np.histogram2d(
            lons, lats, bins=bins, range=[[lon_min, lon_max], [lat_min, lat_max]]
        )
    except ValueError as e:
        print(f"[Skip] {vtk_filename}: histogram2d error - {e}")
        return None

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    Lon, Lat = np.meshgrid(lon_centers, lat_centers)

    # 플롯 시작
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([136, 150, 32, 42], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.STATES, edgecolor="black", facecolor="none")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, color="blue")

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color="gray")
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 10, "color": "black"}
    gl.ylabel_style = {"size": 10, "color": "black"}
    gl.right_labels = False
    gl.top_labels = False

    # 컬러맵 설정
    colors = [
        "#fffffc", "#c0e9fc", "#83c4f1", "#5099cf", "#49a181",
        "#6bbc51", "#69bd50", "#d3e158", "#feaf43", "#f96127",
        "#e1342a", "#9f2b2f", "#891a19"
    ]
    cmap = mcolors.ListedColormap(colors)

    if use_log_scale and H.max() > 1:
        boundaries = np.logspace(0, np.log10(H.max()), len(colors) + 1)
    else:
        boundaries = np.linspace(0, H.max(), len(colors) + 1)

    norm = mcolors.BoundaryNorm(boundaries, ncolors=len(colors), clip=True)

    contour = ax.contourf(
        Lon, Lat, H.T,
        levels=boundaries,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label("Particle Count", fontsize=16)

    # 타임스탬프 추출 및 변환
    match = re.search(r'_(\d{5})\.vtk$', vtk_filename)
    if match:
        time_step = int(match.group(1)) * 100  # 100초 단위
        sim_time = BASE_TIME + datetime.timedelta(seconds=time_step)
        time_str = sim_time.strftime("%B %d, %Y, %H:%M UTC")
    else:
        time_str = "Unknown Time"

    ax.set_title(f"[This Study] Simulation Time: {time_str}", fontsize=20)
    plt.tight_layout()
    return fig

def create_gif_from_vtk_series(vtk_base_path, start, end, step, output_gif="particle_distribution.gif"):
    images = []

    for t in range(start, end + 1, step):
        vtk_filename = f"{vtk_base_path}/Cs-137_{t:05d}.vtk"
        print(f"Processing: {vtk_filename}")
        fig = plot_particle_distribution(vtk_filename)

        if fig is None:
            continue

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
        plt.close(fig)

    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=300, loop=0)
        print(f"[Success] GIF saved as {output_gif}")
    else:
        print("[Warning] No valid frames were generated. GIF not created.")

if __name__ == "__main__":
    create_gif_from_vtk_series("./output_1", start=15, end=1725, step=15)
