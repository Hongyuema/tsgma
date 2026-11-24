from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FOODS = [
    "Red meat",
    "Alcoholic Beverages",
    "Animal fats group",
    "Milk",
    "Eggs",
    "Sugar and Sweeteners",
    "Vegetable Oils",
    "Nuts",
    "Poultry meat",
    "Vegetables",
    "Pulses",
    "Cereals",
    "Fish and seafood",
    "Starchy Roots",
    "Fruit",
    "Other meat",
]
LABELS = {
    "Alcoholic Beverages": "Alcoholic beverages",
    "Animal fats group": "Animal fats",
    "Sugar and Sweeteners": "Sugar and sweeteners",
    "Vegetable Oils": "Vegetable oils",
    "Poultry meat": "Poultry",
    "Cereals": "Selected cereals",
    "Starchy Roots": "Starchy roots",
}
OUTCOMES = ["Non-HDL", "Diabetes1992-2014", "Hypertension"]
PAIRS = [
    ("Animal fats group", "Non-HDL"),
    ("Red meat", "Non-HDL"),
    ("Eggs", "Non-HDL"),
    ("Poultry meat", "Diabetes1992-2014"),
    ("Cereals", "Diabetes1992-2014"),
    ("Sugar and Sweeteners", "Diabetes1992-2014"),
    ("Starchy Roots", "Diabetes1992-2014"),
    ("Alcoholic Beverages", "Hypertension"),
]


def save(fig, directory, name):
    fig.savefig(
        directory / (name + ".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, width, height, text, color="#edf3f5", fontsize=12):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=color,
        edgecolor="#56717c",
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.5,
    )


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": "#56717c", "lw": 1.4},
    )


def run(results, directory, annual_results, diagnostics=None):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    a = pd.read_csv(results / "all_associations.csv")
    fig, ax = plt.subplots(figsize=(8.2, 7.3))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    box(
        ax,
        0.02,
        0.76,
        0.45,
        0.21,
        "Supplied country means\n158 countries or regions\n143 GDP-complete units\n48 food–outcome pairs",
        fontsize=10,
    )
    box(
        ax,
        0.53,
        0.76,
        0.45,
        0.21,
        "Frozen public annual releases\nComplete cereals and official energy\nExplicit annual coverage rules\n140–143 countries per pair",
        fontsize=10,
    )
    box(
        ax,
        0.02,
        0.43,
        0.45,
        0.23,
        "Reference analyses\nCrude and adjusted correlations\nMutual dietary adjustment\nCountry and region exclusions",
        fontsize=10,
    )
    box(
        ax,
        0.53,
        0.43,
        0.45,
        0.23,
        "Reconstructed-data analyses\nCommon 1992–2014 window\nEnergy, rank and regional checks\nCountry and year effects and trends\nFirst differences and pre-2010 fits",
        fontsize=10,
    )
    arrow(ax, (0.245, 0.75), (0.245, 0.67))
    arrow(ax, (0.755, 0.75), (0.755, 0.67))
    box(
        ax,
        0.13,
        0.05,
        0.74,
        0.24,
        "Complete model-specific reporting\nEstimates · 95% intervals · sample sizes · P · BH q\n48 tests within each specification\nSeparate between-country and within-country estimands",
        fontsize=10,
    )
    arrow(ax, (0.245, 0.42), (0.36, 0.30))
    arrow(ax, (0.755, 0.42), (0.64, 0.30))
    save(fig, directory, "Figure_1_workflow")

    cross = pd.read_csv(annual_results / "reconstructed_cross_sectional.csv")
    g = cross[cross.specification.eq("updated_common_window_logGDP_energy")].set_index(
        ["exposure", "outcome"]
    )
    values = np.array([[g.loc[(x, y), "r"] for y in OUTCOMES] for x in FOODS])
    fig, ax = plt.subplots(figsize=(7.5, 8.2))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(
        range(3), ["Non-HDL cholesterol", "Diabetes", "Hypertension"], fontsize=11
    )
    ax.xaxis.tick_top()
    ax.set_yticks(
        range(16),
        ["Complete cereals" if x == "Cereals" else LABELS.get(x, x) for x in FOODS],
        fontsize=11,
    )
    ax.tick_params(length=0, pad=8)
    for i, x in enumerate(FOODS):
        for j, y in enumerate(OUTCOMES):
            row = g.loc[(x, y)]
            q = f"{row.q_BH:.3g}" if row.q_BH < 0.001 else f"{row.q_BH:.3f}"
            color = "white" if abs(row.r) > 0.53 else "#111111"
            ax.text(
                j,
                i,
                f"{row.r:.3f}\nq={q}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Log-GDP-and-energy-adjusted partial correlation")
    fig.tight_layout()
    save(fig, directory, "Figure_2_all_associations")

    diagnostics = diagnostics or annual_results.parent / "diagnostics"
    checks = pd.read_csv(diagnostics / "common_window_diagnostics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.5))
    colors = ["#235b74", "#b47b24", "#785494"]
    markers = ["o", "s", "^"]
    titles = ["Rank-transformed variables", "Region indicators added"]
    for ax, specification, title in zip(
        axes, ["Rank_logGDP_energy", "Region_logGDP_energy"], titles
    ):
        selected = checks[checks.specification.eq(specification)].set_index(
            ["exposure", "outcome"]
        )
        for outcome, color, marker, label in zip(
            OUTCOMES,
            colors,
            markers,
            ["Non-HDL cholesterol", "Diabetes", "Hypertension"],
        ):
            x = [g.loc[(food, outcome), "r"] for food in FOODS]
            y = [selected.loc[(food, outcome), "r"] for food in FOODS]
            ax.scatter(
                x,
                y,
                c=color,
                marker=marker,
                s=36,
                alpha=0.85,
                label=label,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.plot([-0.7, 0.7], [-0.7, 0.7], ls="--", lw=0.9, color="#999999", zorder=0)
        ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
        ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
        ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7))
        ax.set_xlabel("Common-window partial r", fontsize=12.5)
        ax.set_ylabel("Sensitivity partial r", fontsize=12.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, pad=12)
        ax.tick_params(labelsize=12)
    rank = checks[checks.specification.eq("Rank_logGDP_energy")].set_index(
        ["exposure", "outcome"]
    )
    region = checks[checks.specification.eq("Region_logGDP_energy")].set_index(
        ["exposure", "outcome"]
    )
    labels = [
        (
            axes[0],
            rank,
            ("Animal fats group", "Diabetes1992-2014"),
            "Animal fats / diabetes",
            (0, 65),
        ),
        (axes[1], region, ("Milk", "Hypertension"), "Milk / hypertension", (-65, -120)),
        (
            axes[1],
            region,
            ("Starchy Roots", "Non-HDL"),
            "Starchy roots / non-HDL",
            (-40, 95),
        ),
    ]
    for ax, selected, pair, label, offset in labels:
        ax.annotate(
            label,
            (g.loc[pair, "r"], selected.loc[pair, "r"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=11.5,
            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.7},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, directory, "Figure_3_model_sensitivity")

    annual = pd.read_csv(annual_results / "annual_models.csv")
    specs = [
        ("TWFE_logGDP_energy", "Country and year fixed effects", "#235b74", 0.22, "o"),
        (
            "First_difference_logGDP_energy",
            "First differences with year effects",
            "#b47b24",
            0,
            "s",
        ),
        (
            "TWFE_country_trends_logGDP_energy",
            "Fixed effects plus country-specific linear trends",
            "#785494",
            -0.22,
            "^",
        ),
    ]
    for letter, outcome in zip("abc", OUTCOMES):
        fig, ax = plt.subplots(figsize=(8.2, 8.4))
        pairs = [(x, outcome) for x in FOODS]
        for spec, label, color, offset, marker in specs:
            frame = [
                annual[
                    (annual.exposure == x)
                    & (annual.outcome == y)
                    & (annual.specification == spec)
                ].iloc[0]
                for x, y in pairs
            ]
            beta = np.array([row.beta for row in frame])
            low = np.array([row.ci_low for row in frame])
            high = np.array([row.ci_high for row in frame])
            ax.errorbar(
                beta,
                np.arange(len(pairs))[::-1] + offset,
                xerr=[beta - low, high - beta],
                fmt=marker,
                color=color,
                label=label,
                markersize=4.5,
                capsize=2,
                lw=1.2,
            )
        labels = []
        for x, y in pairs:
            row = annual[
                (annual.exposure == x)
                & (annual.outcome == y)
                & (annual.specification == "TWFE_logGDP_energy")
            ].iloc[0]
            label = "Complete cereals" if x == "Cereals" else LABELS.get(x, x)
            labels.append(f"{label}  (n={int(row.n_countries)})")
        ax.set_yticks(np.arange(len(pairs))[::-1], labels, fontsize=11.5)
        ax.axvline(0, color="#777777", lw=0.8)
        ax.grid(axis="x", alpha=0.2)
        ax.set_ylim(-0.5, len(pairs) - 0.5)
        unit = "mmol/L" if outcome == "Non-HDL" else "percentage points"
        title = {
            "Non-HDL": "Non-HDL cholesterol",
            "Diabetes1992-2014": "Diabetes",
            "Hypertension": "Hypertension",
        }[outcome]
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel(
            f"Outcome difference ({unit}) per 100 kcal/person/day", fontsize=10.5
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.45, -0.11),
            frameon=False,
            fontsize=10.5,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        save(fig, directory, f"Figure_4{letter}_within_country")

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    box(
        ax,
        0.02,
        0.25,
        0.27,
        0.64,
        "Frozen annual reconstruction\nCommon 1992–2014 window\n140–143 countries per pair\n\n16 foods × 3 health outcomes\nDefined food categories\nand total energy",
        fontsize=10,
    )
    box(
        ax,
        0.365,
        0.25,
        0.27,
        0.64,
        "Between-country comparisons\nLog GDP and energy adjusted\nAt within-model BH q<0.05\n15 with conventional inference\n11 with HC3 inference\n9 with region indicators added",
        fontsize=10,
    )
    box(
        ax,
        0.71,
        0.25,
        0.27,
        0.64,
        "Within-country, 1992–2014\nLog GDP and energy adjusted\nAt within-model BH q<0.05\n9 with country and year effects\n2 with first differences\n0 with added country trends",
        fontsize=10,
    )
    ax.text(
        0.5,
        0.10,
        "Each model evaluates all 48 pairs with its own testing family and estimand.",
        ha="center",
        va="center",
        fontsize=10,
    )
    save(fig, directory, "Graphical_abstract")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annual-results", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    run(args.results, args.output, args.annual_results, args.diagnostics)
