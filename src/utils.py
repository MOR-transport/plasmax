"""
Figure export utility: saves matplotlib figures as PNG and PGFPlots/TikZ.

Replaces the outdated tikzplotlib package. The TikZ export is inspired by
the MATLAB matlab2tikz package (https://github.com/matlab2tikz/matlab2tikz).

Public API
----------
``save_fig(filename, fig, ...)``
    Save *fig* as a raster image **and** a ``.tex`` file in one call — works
    like ``fig.savefig()`` but also writes PGFPlots/TikZ output alongside it.

    Supported plot types:

    * Line plots (``plot``, ``semilogy``, ``loglog``, …) → ``\\addplot``
    * Scatter plots → ``\\addplot ... only marks``
    * Bar plots (``bar``, incl. stacked via ``bottom=``) → ``\\addplot ... ybar``
    * Text annotations (``ax.text``) → ``\\node`` at the matching ``axis cs``
    * 2-D colour maps (``pcolormesh``, ``imshow``) → PNG embedded via
      ``\\addplot graphics``
    * Colorbars → PGFPlots ``colorbar`` axis option
    * Multiple subplots → each axis sized/positioned as a fraction of
      *width*/*height* (default ``\\figurewidth``/``\\figureheight``), so the
      whole panel grid rescales with those lengths like a single-axes plot
    * Colormaps unknown to PGFPlots (e.g. ``turbo``) → auto-defined via
      ``\\pgfplotsset{colormap={name}{...}}``

Typical usage
-------------
    from src.utils import save_fig

    fig, ax = plt.subplots()
    ax.plot(x, y, label=r"$f(x)$")
    save_fig("results/plot.png", fig)   # writes plot.png  and  plot.tex

Required LaTeX preamble
-----------------------
    \\usepackage{pgfplots}
    \\pgfplotsset{compat=newest}
    \\usepgfplotslibrary{colormaps}
    \\newlength{\\figurewidth}  \\setlength{\\figurewidth}{8cm}
    \\newlength{\\figureheight} \\setlength{\\figureheight}{6cm}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.container as mcontainer
import matplotlib.lines as mlines
import matplotlib.image as mimage
import numpy as np


# ---------------------------------------------------------------------------
# Style translation tables (matplotlib → PGFPlots)
# ---------------------------------------------------------------------------

_LS_MAP: dict[str, str] = {
    "solid":    "",
    "-":        "",
    "dashed":   "dashed",
    "--":       "dashed",
    "dotted":   "dotted",
    ":":        "dotted",
    "dashdot":  "dash dot",
    "-.":       "dash dot",
    "None":     "",
    "none":     "",
    "":         "",
}

_MARK_MAP: dict[str, str] = {
    "o":    "o",
    "s":    "square",
    "^":    "triangle",
    "v":    "triangle*",
    "<":    "triangle",
    ">":    "triangle",
    "D":    "diamond",
    "d":    "diamond*",
    "p":    "pentagon*",
    "*":    "asterisk",
    "+":    "+",
    "x":    "x",
    ".":    "*",
    ",":    "*",
    "None": "",
    "none": "",
    "":     "",
}

_LOC_MAP: dict[int, str] = {
    # All standard matplotlib locations are *inside* the axes.
    # 0 = "best" → omit legend pos so PGFPlots auto-places inside (key missing)
    1:  "north east",
    2:  "north west",
    3:  "south west",
    4:  "south east",
    5:  "east",           # "right"
    6:  "west",           # "center left"
    7:  "east",           # "center right"
    8:  "south",          # "lower center"
    9:  "north",          # "upper center"
    10: "center",
}

# PGFPlots built-in colormaps (via \usepgfplotslibrary{colormaps})
_KNOWN_PGF_CMAPS: frozenset[str] = frozenset({
    "viridis", "plasma", "inferno", "magma", "cividis",
    "hot", "cool", "jet", "bone", "copper",
    "blackwhite", "greenyellow", "redyellow", "bluered",
    "hot2", "cool2",
})


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _color_key(c) -> tuple[float, float, float]:
    r, g, b, _ = mcolors.to_rgba(c)
    return (round(r, 5), round(g, 5), round(b, 5))


def _get_cname(c, color_defs: dict) -> str:
    """Return a unique PGFPlots color name for c, registering it if new."""
    key = _color_key(c)
    if key not in color_defs:
        color_defs[key] = f"mplcolor{len(color_defs)}"
    return color_defs[key]


def _color_defs_tex(color_defs: dict) -> str:
    return "\n".join(
        f"\\definecolor{{{name}}}{{rgb}}{{{r:.6f},{g:.6f},{b:.6f}}}"
        for (r, g, b), name in color_defs.items()
    )


def _pgf_colormap_name(cmap, extra_cmaps: dict) -> str:
    """
    Return the PGFPlots colormap name for cmap.
    Colormaps unknown to PGFPlots are registered in *extra_cmaps* for
    later definition via ``\\pgfplotsset{colormap=...}``.
    """
    name = getattr(cmap, "name", str(cmap))
    # Normalise common aliases
    alias = {"gray": "blackwhite", "grey": "blackwhite", "turbo": "turbo"}
    pgf_name = alias.get(name, name)
    if pgf_name not in _KNOWN_PGF_CMAPS and pgf_name not in extra_cmaps:
        extra_cmaps[pgf_name] = cmap
    return pgf_name


def _define_pgf_colormap(name: str, cmap, n: int = 64) -> str:
    """Generate a ``\\pgfplotsset{colormap={name}{...}}`` definition."""
    xs = np.linspace(0.0, 1.0, n)
    rgb255 = (cmap(xs)[:, :3] * 255).round().astype(int)
    entries = " ".join(
        f"rgb255({i / (n - 1):.4f})=({r},{g},{b})"
        for i, (r, g, b) in enumerate(rgb255)
    )
    return f"\\pgfplotsset{{colormap={{{name}}}{{{entries}}}}}"


# ---------------------------------------------------------------------------
# Text escaping
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """Pass LaTeX math/commands through; escape bare special characters."""
    if not text:
        return ""
    if "$" in text or "\\" in text:
        return text
    for src, dst in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        text = text.replace(src, dst)
    return text


# ---------------------------------------------------------------------------
# Line plot → \addplot
# ---------------------------------------------------------------------------

def _line_addplot(
    line: mlines.Line2D,
    color_defs: dict,
    fmt: str,
    max_chunk: int,
    forget: bool,
) -> str:
    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)
    if len(x) == 0:
        return ""

    opts: list[str] = []

    opts.append(f"color={_get_cname(line.get_color(), color_defs)}")

    ls = line.get_linestyle()
    ls_pgf = _LS_MAP.get(ls, _LS_MAP.get(str(ls), ""))
    if ls_pgf:
        opts.append(ls_pgf)

    mk = str(line.get_marker())
    mk_pgf = _MARK_MAP.get(mk, "")
    opts.append(f"mark={mk_pgf}" if mk_pgf else "mark=none")

    no_line = ls in ("None", "none", "") or not ls
    if no_line and mk_pgf:
        opts.append("only marks")

    lw = line.get_linewidth()
    if abs(lw - 1.5) > 0.05:
        opts.append(f"line width={lw:.2f}pt")

    if forget:
        opts.append("forget plot")

    opts_str = ", ".join(opts)

    parts: list[str] = []
    n = len(x)
    for start in range(0, n, max_chunk):
        end = min(start + max_chunk, n)
        coords = " ".join(
            f"({fmt % float(xi)},{fmt % float(yi)})"
            for xi, yi in zip(x[start:end], y[start:end])
            if np.isfinite(xi) and np.isfinite(yi)
        )
        if coords:
            parts.append(
                f"\\addplot [{opts_str}]\n"
                f"  coordinates {{{coords}}};\n"
            )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Raster image helpers (pcolormesh / imshow → PNG)
# ---------------------------------------------------------------------------

def _quadmesh_png(
    mesh: mcollections.QuadMesh, path: Path
) -> tuple[float, float, float, float]:
    """Render a QuadMesh to a PNG file; return its (xmin, xmax, ymin, ymax)."""
    coords = mesh._coordinates  # shape (nrows+1, ncols+1, 2)
    nr = coords.shape[0] - 1
    nc = coords.shape[1] - 1

    arr = np.asarray(mesh.get_array(), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(nr, nc)

    rgba = mesh.get_cmap()(mesh.norm(arr))
    rgba = np.clip(rgba, 0.0, 1.0)
    # pcolormesh row 0 is at the bottom; PNG row 0 is at the top
    rgba = rgba[::-1]

    import matplotlib.pyplot as _plt
    _plt.imsave(str(path), rgba)

    xmin = float(coords[:, :, 0].min())
    xmax = float(coords[:, :, 0].max())
    ymin = float(coords[:, :, 1].min())
    ymax = float(coords[:, :, 1].max())
    return xmin, xmax, ymin, ymax


def _aximage_png(
    im: mimage.AxesImage, path: Path
) -> tuple[float, float, float, float]:
    """Render an AxesImage (imshow) to a PNG file; return its extent."""
    arr = np.asarray(im.get_array())

    if arr.ndim == 2:
        rgba = im.get_cmap()(im.norm(arr.astype(float)))
        rgba = np.clip(rgba, 0.0, 1.0)
    else:
        rgba = arr

    import matplotlib.pyplot as _plt
    _plt.imsave(str(path), rgba)

    ext = im.get_extent()
    if ext is None:
        h, w = arr.shape[:2]
        ext = [-0.5, w - 0.5, -0.5, h - 0.5]
    return float(ext[0]), float(ext[1]), float(ext[2]), float(ext[3])


# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------

def _scale_len(frac: float, base: str) -> str:
    """Express *frac* of the LaTeX length *base* (e.g. ``"\\figurewidth"`` or
    a literal ``"8cm"``) as a TeX-valid length string.

    For a literal ``cm`` length this multiplies the numeric value; for a
    macro length (``\\figurewidth``, ``\\figureheight``, ...) this relies on
    plain TeX allowing a decimal multiplier directly in front of a length,
    e.g. ``0.1234\\figurewidth``. *base* may itself already be a scaled
    macro length (``"0.1839\\figurewidth"``, as produced by a previous call)
    -- the leading factor is peeled off and folded into *frac* so scale
    factors compose instead of being nested/concatenated.
    """
    if base.endswith("cm"):
        return f"{frac * float(base[:-2]):.4f}cm"
    m = re.match(r"^([0-9.]+)(\\.+)$", base)
    if m:
        frac *= float(m.group(1))
        base = m.group(2)
    return f"{frac:.4f}{base}"


def _has_grid(ax: matplotlib.axes.Axes, axis: str) -> bool:
    try:
        lines = ax.get_xgridlines() if axis == "x" else ax.get_ygridlines()
        return bool(lines) and any(l.get_visible() for l in lines)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Colorbar detection
# ---------------------------------------------------------------------------

def _get_mappable_colorbar(mappable):
    """Return the Colorbar object attached to a mappable, or None."""
    return getattr(mappable, "colorbar", None)


# ---------------------------------------------------------------------------
# Single-axes block builder
# ---------------------------------------------------------------------------

def _axes_block(
    ax: matplotlib.axes.Axes,
    color_defs: dict,
    extra_cmaps: dict,
    fmt: str,
    max_chunk: int,
    png_dir: Path,
    png_stem: str,
    png_count: list,  # [int] mutable counter shared across axes
    extra_axis_opts: list[str],
    width: str,
    height: str,
    at: Optional[str] = None,
) -> str:
    """Return a complete ``\\begin{axis}...\\end{axis}`` block."""

    axis_opts: list[str] = []

    if at:
        axis_opts.append(f"at={{{at}}}")
        axis_opts.append("anchor=south west")

    axis_opts.append(f"width={width}")
    axis_opts.append(f"height={height}")

    # Axis scales
    if ax.get_xscale() == "log":
        axis_opts.append("xmode=log, log basis x=10")
    if ax.get_yscale() == "log":
        axis_opts.append("ymode=log, log basis y=10")

    # Labels / title
    if xl := ax.get_xlabel():
        axis_opts.append(f"xlabel={{{_escape(xl)}}}")
    if yl := ax.get_ylabel():
        axis_opts.append(f"ylabel={{{_escape(yl)}}}")
    if tt := ax.get_title():
        axis_opts.append(f"title={{{_escape(tt)}}}")

    # Axis limits
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    axis_opts.append(f"xmin={fmt % x0}, xmax={fmt % x1}")
    axis_opts.append(f"ymin={fmt % y0}, ymax={fmt % y1}")

    # Grid
    if _has_grid(ax, "x"):
        axis_opts.append("xmajorgrids=true")
    if _has_grid(ax, "y"):
        axis_opts.append("ymajorgrids=true")

    # Colormap / point meta from the first ScalarMappable child
    cmap_registered = False
    for mappable in list(ax.collections) + list(ax.images):
        if not isinstance(mappable, (mcollections.QuadMesh, mimage.AxesImage)):
            continue
        if not mappable.get_visible():
            continue
        if cmap_registered:
            break
        clim = mappable.get_clim()
        if clim[0] is not None:
            axis_opts.append(f"point meta min={fmt % clim[0]}")
            axis_opts.append(f"point meta max={fmt % clim[1]}")
        pgf_cmap = _pgf_colormap_name(mappable.get_cmap(), extra_cmaps)
        axis_opts.append(f"colormap name={pgf_cmap}")
        cbar = _get_mappable_colorbar(mappable)
        if cbar is not None:
            axis_opts.append("colorbar")
            # Scale colorbar width proportional to the axis width so it
            # always looks right regardless of the \figurewidth chosen.
            frac = cbar.ax.get_position().width / ax.get_position().width
            cbar_w = _scale_len(frac, width)
            axis_opts.append(f"colorbar style={{width={cbar_w}}}")
        cmap_registered = True

    # Extra user-supplied options
    axis_opts.extend(extra_axis_opts)

    # Legend: collect labels and location
    legend = ax.get_legend()
    legend_labels: set[str] = set()
    if legend and legend.get_visible():
        for t in legend.get_texts():
            legend_labels.add(t.get_text())
        loc = int(getattr(legend, "_loc_real", getattr(legend, "_loc", 0)))
        if loc in _LOC_MAP:
            axis_opts.append(f"legend pos={_LOC_MAP[loc]}")

    # Body -------------------------------------------------------------------
    body: list[str] = []
    legend_entries: list[str] = []

    # Line plots
    for line in ax.get_lines():
        if not line.get_visible():
            continue
        label = line.get_label()
        in_legend = bool(label and not label.startswith("_") and label in legend_labels)
        code = _line_addplot(line, color_defs, fmt, max_chunk, forget=not in_legend)
        if code:
            body.append(code)
            if in_legend:
                legend_entries.append(f"\\addlegendentry{{{_escape(label)}}}\n")

    # Collections (pcolormesh → QuadMesh, scatter → PathCollection)
    for coll in ax.collections:
        if not coll.get_visible():
            continue

        if isinstance(coll, mcollections.QuadMesh):
            png_count[0] += 1
            png_name = f"{png_stem}_{png_count[0]:03d}.png"
            xmn, xmx, ymn, ymx = _quadmesh_png(coll, png_dir / png_name)
            body.append(
                f"\\addplot graphics [\n"
                f"  xmin={fmt % xmn}, xmax={fmt % xmx},\n"
                f"  ymin={fmt % ymn}, ymax={fmt % ymx}\n"
                f"] {{{png_name}}};\n"
            )

        elif isinstance(coll, mcollections.PathCollection):
            offsets = coll.get_offsets()   # masked array, shape (N, 2)
            if len(offsets) == 0:
                continue
            pts = np.asarray(offsets)

            # Color: use first face colour (single-colour scatter).
            # Explicitly set both color (draw) and fill so the legend icon
            # matches the markers — PGFPlots does not automatically fill
            # mark=* from the draw color alone.
            fc = coll.get_facecolors()
            cname = _get_cname(fc[0] if len(fc) else "C0", color_defs)

            # Marker size: matplotlib s is area in pt², pgfplots mark size is radius.
            # Use at least 2pt so the legend icon stays clearly visible.
            sizes = coll.get_sizes()
            mark_size = max(float(np.sqrt(sizes[0])) / 2 if len(sizes) else 2.0, 2.0)

            opts = [
                f"color={cname}",
                "only marks",
                "mark=*",
                f"mark size={mark_size:.2f}pt",
                f"mark options={{fill={cname}, draw={cname}}}",
            ]

            alpha = coll.get_alpha()
            if alpha is not None and alpha < 1.0:
                opts.append(f"opacity={alpha:.2f}")

            label = coll.get_label()
            in_legend = bool(label and not label.startswith("_") and label in legend_labels)
            if not in_legend:
                opts.append("forget plot")

            coords = " ".join(
                f"({fmt % float(xi)},{fmt % float(yi)})"
                for xi, yi in pts
                if np.isfinite(xi) and np.isfinite(yi)
            )
            body.append(f"\\addplot [{', '.join(opts)}]\n  coordinates {{{coords}}};\n")
            if in_legend:
                legend_entries.append(f"\\addlegendentry{{{_escape(label)}}}\n")

    # Bar plots (ax.bar() → BarContainer of Rectangle patches). Multiple
    # containers are assumed to be a stacked chart (ax.bar(..., bottom=...)),
    # since that's the only multi-container case this handles; each addplot
    # gets its own (unstacked) height and pgfplots' "ybar stacked" does the
    # summing, so ungrouped/dodged multi-container bars are not supported.
    bar_containers = [c for c in ax.containers
                       if isinstance(c, mcontainer.BarContainer) and c.patches]
    if bar_containers:
        axis_opts.append("ybar")
        if len(bar_containers) > 1:
            axis_opts.append("ybar stacked")

        bar_w_data = bar_containers[0].patches[0].get_width()
        bar_w_frac = bar_w_data / (x1 - x0) if (x1 - x0) else 0.0
        axis_opts.append(f"bar width={_scale_len(bar_w_frac, width)}")

        xticks = ax.get_xticks()
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        if len(xticks) and any(xticklabels):
            axis_opts.append(f"xtick={{{','.join(fmt % xt for xt in xticks)}}}")
            axis_opts.append(f"xticklabels={{{','.join(_escape(t) for t in xticklabels)}}}")

        for container in bar_containers:
            patches = container.patches
            cname = _get_cname(patches[0].get_facecolor(), color_defs)
            coords = " ".join(
                f"({fmt % (p.get_x() + p.get_width() / 2)},{fmt % p.get_height()})"
                for p in patches
            )
            label = container.get_label()
            in_legend = bool(label and not label.startswith("_") and label in legend_labels)
            opts = [f"color={cname}", f"fill={cname}"]
            if not in_legend:
                opts.append("forget plot")
            body.append(f"\\addplot [{', '.join(opts)}] coordinates {{{coords}}};\n")
            if in_legend:
                legend_entries.append(f"\\addlegendentry{{{_escape(label)}}}\n")

    # Standalone text annotations (ax.text(), e.g. bar-value labels)
    for txt in ax.texts:
        if not txt.get_visible():
            continue
        content = txt.get_text()
        if not content:
            continue
        tx, ty = txt.get_position()
        anchor_v = {"bottom": "south", "top": "north", "baseline": "south"}.get(txt.get_va(), "")
        anchor_h = {"left": "west", "right": "east"}.get(txt.get_ha(), "")
        anchor = " ".join(p for p in (anchor_v, anchor_h) if p) or "center"
        cname = _get_cname(txt.get_color(), color_defs)
        body.append(
            f"\\node[anchor={anchor}, text={cname}] at "
            f"(axis cs:{fmt % tx},{fmt % ty}) {{{_escape(content)}}};\n"
        )

    # Images (imshow → AxesImage)
    for im in ax.images:
        if not im.get_visible():
            continue
        png_count[0] += 1
        png_name = f"{png_stem}_{png_count[0]:03d}.png"
        xmn, xmx, ymn, ymx = _aximage_png(im, png_dir / png_name)
        body.append(
            f"\\addplot graphics [\n"
            f"  xmin={fmt % xmn}, xmax={fmt % xmx},\n"
            f"  ymin={fmt % ymn}, ymax={fmt % ymx}\n"
            f"] {{{png_name}}};\n"
        )

    body.extend(legend_entries)

    opts_str = ",\n  ".join(axis_opts)
    return (
        f"\\begin{{axis}}[\n  {opts_str}\n]\n"
        + "".join(body)
        + "\\end{axis}\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PREAMBLE = """\
\\usepackage{pgfplots}
\\pgfplotsset{compat=newest}
\\usepgfplotslibrary{colormaps}
\\usetikzlibrary{plotmarks, arrows.meta}
\\usepackage{amsmath}"""


def _to_tikz(
    tex_path: Path,
    fig: matplotlib.figure.Figure,
    *,
    width: str = r"\figurewidth",
    height: str = r"\figureheight",
    standalone: bool = False,
    extra_axis_parameters: Optional[list[str]] = None,
    extra_tikzpicture_options: Optional[list[str]] = None,
    float_format: str = "%.6g",
    max_chunk_length: int = 4000,
    show_info: bool = True,
) -> None:
    """Write only the TikZ/PGFPlots ``.tex`` file for *fig*."""
    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    png_stem = tex_path.stem
    png_dir = tex_path.parent
    extra_axis_opts = list(extra_axis_parameters or [])
    extra_tikz_opts = list(extra_tikzpicture_options or [])
    png_count: list[int] = [0]
    color_defs: dict = {}
    extra_cmaps: dict = {}

    fig.canvas.draw()

    colorbar_ax_ids: set[int] = set()
    for ax in fig.get_axes():
        for mappable in list(ax.collections) + list(ax.images):
            cbar = getattr(mappable, "colorbar", None)
            if cbar is not None:
                colorbar_ax_ids.add(id(cbar.ax))

    data_axes = [ax for ax in fig.get_axes() if id(ax) not in colorbar_ax_ids]
    if not data_axes:
        raise ValueError("save_fig: no data axes found in figure.")

    single = len(data_axes) == 1
    axes_blocks: list[str] = []

    for ax in data_axes:
        if single:
            at_str = None
            w_str, h_str = width, height
        else:
            # Scale the whole multi-panel figure to width x height (default
            # \figurewidth x \figureheight) and place each axis at its
            # matplotlib-layout fraction of that box, so the figure resizes
            # with the including document's chosen \figurewidth/\figureheight
            # instead of being pinned to its original absolute inches.
            pos = ax.get_position()
            at_str = f"({_scale_len(pos.x0, width)}, {_scale_len(pos.y0, height)})"
            w_str = _scale_len(pos.width, width)
            h_str = _scale_len(pos.height, height)

        axes_blocks.append(_axes_block(
            ax, color_defs, extra_cmaps, float_format, max_chunk_length,
            png_dir, png_stem, png_count, extra_axis_opts,
            w_str, h_str, at=at_str,
        ))

    header_parts: list[str] = []
    color_tex = _color_defs_tex(color_defs)
    if color_tex:
        header_parts.append(color_tex)
    for cmap_name, cmap_obj in extra_cmaps.items():
        header_parts.append(_define_pgf_colormap(cmap_name, cmap_obj))
    header = "\n".join(header_parts)
    if header:
        header += "\n"

    tikz_opt_str = f"[{', '.join(extra_tikz_opts)}]" if extra_tikz_opts else ""
    body = (
        header
        + f"\\begin{{tikzpicture}}{tikz_opt_str}\n"
        + "\n".join(axes_blocks)
        + "\\end{tikzpicture}\n"
    )

    if standalone:
        body = (
            "\\documentclass[tikz]{standalone}\n"
            + _PREAMBLE + "\n"
            + "\\begin{document}\n"
            + body
            + "\\end{document}\n"
        )

    tex_path.write_text(body, encoding="utf-8")

    if show_info:
        raster_msg = f" ({png_count[0]} raster image(s) alongside)" if png_count[0] else ""
        print(f"save_fig: '{tex_path}' written{raster_msg}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_fig(
    filename: str | Path,
    fig: Optional[matplotlib.figure.Figure] = None,
    *,
    dpi: int = 150,
    bbox_inches: str = "tight",
    width: str = r"\figurewidth",
    height: str = r"\figureheight",
    standalone: bool = False,
    extra_axis_parameters: Optional[list[str]] = None,
    extra_tikzpicture_options: Optional[list[str]] = None,
    float_format: str = "%.6g",
    max_chunk_length: int = 4000,
    show_info: bool = True,
) -> None:
    """
    Save a matplotlib figure as a PNG image **and** a PGFPlots/TikZ file.

    Works like ``fig.savefig(filename)`` but also writes a ``.tex`` file so
    the plot can be included directly in a LaTeX document.  Both files share
    the same stem; the extension of *filename* controls only the image format.

    Parameters
    ----------
    filename : str or Path
        Output path.  The extension sets the image format (default ``.png``
        when no extension is given).  A ``.tex`` file is always written
        alongside using the same stem.
    fig : Figure, optional
        Figure to export.  Defaults to ``plt.gcf()``.
    dpi : int
        Resolution for the raster image (default 150).
    bbox_inches : str
        Passed to ``fig.savefig`` (default ``'tight'``).
    width, height : str
        LaTeX lengths for the TikZ axis dimensions.  Define them in your
        document preamble::

            \\newlength{\\figurewidth}\\setlength{\\figurewidth}{8cm}
    standalone : bool
        Wrap the TikZ output in a compilable ``standalone`` LaTeX document.
    extra_axis_parameters : list[str], optional
        Extra PGFPlots options appended to every ``\\begin{axis}[...]``.
    extra_tikzpicture_options : list[str], optional
        Extra options for ``\\begin{tikzpicture}[...]``.
    float_format : str
        Python format string for numeric values (default ``'%.6g'``).
    max_chunk_length : int
        Max coordinate pairs per ``\\addplot coordinates`` block (avoids
        pdfTeX buffer overflow on very long data series).
    show_info : bool
        Print filenames to stdout when done.
    """
    if fig is None:
        import matplotlib.pyplot as _plt
        fig = _plt.gcf()

    path = Path(filename)
    # Default to .png when no extension provided
    if not path.suffix:
        path = path.with_suffix(".png")

    stem = path.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    # --- raster image (mimics fig.savefig) ---
    fig.savefig(str(path), dpi=dpi, bbox_inches=bbox_inches)
    if show_info:
        print(f"save_fig: '{path}' written.")

    # --- TikZ / PGFPlots ---
    _to_tikz(
        stem.with_suffix(".tex"),
        fig,
        width=width,
        height=height,
        standalone=standalone,
        extra_axis_parameters=extra_axis_parameters,
        extra_tikzpicture_options=extra_tikzpicture_options,
        float_format=float_format,
        max_chunk_length=max_chunk_length,
        show_info=show_info,
    )
