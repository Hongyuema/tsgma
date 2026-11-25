"""Prepare the methodological submission panels using the supplied artwork and audited estimates."""

from pathlib import Path
from shutil import copyfile
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

root = Path.cwd()
assets = root / "artwork_sources"
out = root / "figures"
font = font_manager.findfont("DejaVu Sans")


def label(im, box, message, size=13, fill="white"):
    w, h = im.size
    x0, y0, x1, y1 = [round(v * (w if j % 2 == 0 else h)) for j, v in enumerate(box)]
    d = ImageDraw.Draw(im)
    d.rectangle((x0, y0, x1, y1), fill=fill)
    f = ImageFont.truetype(font, max(10, round(size * w / 640)))
    d.multiline_text(
        ((x0 + x1) / 2, (y0 + y1) / 2),
        message,
        font=f,
        anchor="mm",
        align="center",
        fill="#202538",
    )


def snapshot(im, name):
    im.save(out / (name + ".png"))
    fig = plt.figure(figsize=(7, 7 * im.height / im.width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im, interpolation="none")
    ax.axis("off")
    fig.savefig(out / (name + ".pdf"))
    plt.close(fig)


im = Image.open(assets / "abstract.png").convert("RGB")
label(
    im, (0.36, 0.901, 0.621, 0.959), "Quantitative lag model\n(previous 1–5 years)", 10
)
label(
    im,
    (0.647, 0.063, 0.986, 0.578),
    "Third-star decision\n\nCrude, adjusted and delayed\nassociations in the same\ndirection\n\n0 of 48 pairs met the rule",
    11,
)
snapshot(im, "Graphical_abstract")
from cumulative_workflow import draw as draw_cumulative_workflow

draw_cumulative_workflow(assets / "flow.png", out / "Figure_1")
im = Image.open(assets / "Original_network.png").convert("RGB")
snapshot(im, "Figure_5")
copyfile(assets / "Original_network.png", out / "Figure_5.png")
tab = pd.read_csv(root / "corrected_results/all_associations.csv")
snap = pd.read_csv(root / "outcome_audit/corrected_snapshot.csv")
plot = pd.read_csv(root / "historical_plot_audit/plot_year_values.csv")
for idx, (o, colors) in enumerate(
    zip(
        ["Non-HDL", "Diabetes1992-2014", "Hypertension"],
        [("#976ee2", "#68cf84"), ("#ffa330", "#ef65c6"), ("#ff6865", "#4caff3")],
    ),
    2,
):
    for tag, spec, color in zip("ac", ["Pearson_158", "GDP_partial_143"], colors):
        rows = tab[(tab.specification == spec) & (tab.outcome == o)]
        assert len(rows) == 16
        names = rows.exposure.tolist()
        theta = np.arange(16) * 2 * np.pi / 16
        values = rows.r.to_numpy() + 1
        fig = plt.figure(figsize=(7, 5.5))
        ax = fig.add_axes([0.21, 0.15, 0.58, 0.70], projection="polar")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 2)
        labs = [
            x.replace("Beverages", "beverages")
            .replace("Animal fats group", "Animal fats")
            .replace("Sugar and Sweeteners", "Sugar / sweeteners")
            for x in names
        ]
        ax.set_xticks(theta, labs, fontsize=8)
        for t, l in zip(theta, ax.get_xticklabels()):
            l.set_ha(
                "left"
                if 0.15 < t < np.pi - 0.15
                else "right" if t > np.pi + 0.15 else "center"
            )
        ax.set_yticks([0.5, 1, 1.5, 2], ["−0.5", "0", "0.5", "1"], fontsize=7)
        ax.grid(alpha=0.35)
        ax.plot(np.r_[theta, theta[0]], np.r_[values, values[0]], color=color)
        ax.fill(theta, values, color=color, alpha=0.19)
        ax.scatter(
            theta,
            values,
            facecolors=[color if v < 0.05 else "white" for v in rows.q_BH],
            edgecolors=color,
            zorder=4,
        )
        fig.text(
            0.5,
            0.04,
            "Filled points meet BH q < 0.05 across 48 pairs.",
            ha="center",
            fontsize=9,
        )
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"Submitted_{idx}{tag}.{ext}", dpi=220)
        plt.close(fig)
    if idx == 3:
        snapshot(
            Image.open(assets / "Historical_diabetes_scatter.png"), f"Submitted_{idx}b"
        )
    else:
        fig, axes = plt.subplots(4, 4, figsize=(8, 7.5), layout="constrained")
        for food, ax in zip(names, axes.flat):
            x = snap[food]
            y = snap[o]
            ax.scatter(x, y, s=4, alpha=0.5, color=colors[0])
            order = np.sort(x)
            ax.plot(
                order, np.polyval(np.polyfit(x, y, 1), order), color=colors[1], lw=1.4
            )
            ax.set_title(food.replace("Animal fats group", "Animal fats"), fontsize=8)
            ax.text(
                0.96,
                0.90,
                f"r²={x.corr(y)**2:.3f}",
                transform=ax.transAxes,
                ha="right",
                fontsize=7,
            )
            ax.tick_params(labelsize=6)
        fig.supxlabel("National dietary availability (kcal/person/day)", fontsize=9)
        fig.supylabel(
            (
                "Non-HDL cholesterol (mmol/L)"
                if idx == 2
                else "Hypertension prevalence (%)"
            ),
            fontsize=9,
        )
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"Submitted_{idx}b.{ext}", dpi=220)
        plt.close(fig)
    im = Image.open(assets / f"Historical_{idx}d.png").convert("RGB")
    if idx == 4:
        # The original UK plot extends past the source. The Saudi and Turkish
        # health curves were assigned to other countries. Replace these tiles only.
        for code, title, box in [
            ("GBR", "United Kingdom", (231, 757, 381, 834)),
            ("SAU", "Saudi Arabia", (538, 833, 698, 910)),
            ("TUR", "Turkey", (383, 910, 537, 990)),
        ]:
            z = plot[
                (plot.exposure == "Alcoholic beverages") & (plot.Code == code)
            ].sort_values("Year")
            fig, ax = plt.subplots(figsize=(3.05, 1.54))
            fig.subplots_adjust(left=0.21, right=0.80, top=0.92, bottom=0.34)
            ax2 = ax.twinx()
            for a, col, color in [
                (ax, "plot_food", "#ef6464"),
                (ax2, "source_health", "#5babe8"),
            ]:
                v = z.dropna(subset=[col])
                if col == "source_health":
                    v = v[v.Year.between(1990, 2019)]
                a.fill_between(v.Year, v[col], v[col].min(), alpha=0.35, color=color)
                a.plot(v.Year, v[col], lw=0.9, color=color)
                a.tick_params(labelsize=6, pad=1)
            ax.set_ylabel("Alcohol (kcal)", fontsize=6)
            ax2.set_ylabel("Hypertension (%)", fontsize=6)
            ax.set_xlabel(title, fontsize=7, labelpad=1)
            fig.canvas.draw()
            tile = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
            l, t, r, b = box
            coords = tuple(
                round((v - (77 if j % 2 == 0 else 472)) * 4 / 1.3)
                for j, v in enumerate(box)
            )
            im.paste(
                tile.resize(
                    (coords[2] - coords[0], coords[3] - coords[1]),
                    Image.Resampling.LANCZOS,
                ),
                coords[:2],
            )
            plt.close(fig)
    snapshot(im, f"Submitted_{idx}d")
