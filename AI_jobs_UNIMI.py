import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import os
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
PRIMARY = "#2F6B9A"
SECONDARY = "#E08A24"
ACCENT = "#C74432"
POSITIVE = "#3A7D44"
NEUTRAL = "#B8B8B8"
LIGHT_NEUTRAL = "#E6E6E6"
DARK = "#2B2B2B"
GRID = "#D9D9D9"

plt.rcParams.update({
    "font.family": ["Segoe UI", "Calibri", "Arial", "sans-serif"],
    "font.size": 12,
    "font.weight": "regular",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "text.color": DARK,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

sns.set_theme(
    style="whitegrid",
    rc={
        "font.family": ["Segoe UI", "Calibri", "Arial", "sans-serif"],
        "axes.edgecolor": "#BFBFBF",
        "grid.color": GRID,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8
    }
)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'grafici_esame')
os.makedirs(output_dir, exist_ok=True)

import kagglehub

def load_data():

    path = kagglehub.dataset_download(
        "khushikyad001/ai-impact-on-jobs-2030"
    )

    files = os.listdir(path)
    csv_file = [f for f in files if f.endswith(".csv")][0]

    return pd.read_csv(os.path.join(path, csv_file))

def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches="tight", facecolor="white")
    plt.close()

def style_axes(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.8, alpha=0.35, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BFBFBF")
    ax.spines["bottom"].set_color("#BFBFBF")
    ax.tick_params(axis="both", labelsize=11, colors=DARK)
    ax.title.set_fontweight("bold")
    return ax

def style_legend(ax, outside=True, title=None):
    if outside:
        return ax.legend(title=title, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=11)
    return ax.legend(title=title, frameon=False, fontsize=11)

def plot_grafico_1(df):
    top_n = 10
    top_jobs = (
        df.groupby("Job_Title", as_index=False)["Automation_Probability_2030"]
          .mean()
          .sort_values("Automation_Probability_2030", ascending=False)
          .head(top_n)
    )
    plt.figure(figsize=(10, 6))

    colors = sns.color_palette("Reds_r", n_colors=len(top_jobs))

    ax = sns.barplot(
        data=top_jobs,
        x="Automation_Probability_2030",
        y="Job_Title",
        hue="Job_Title",
        palette=colors,
        dodge=False,
        legend=False
    )

    ax.set_title("Lavori più esposti all'automazione", fontsize=14, weight="bold")
    ax.set_xlabel("Probabilità di automazione (media)")
    ax.set_ylabel("")

    for i, v in enumerate(top_jobs["Automation_Probability_2030"]):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=10)

    plt.xlim(0, 1)
    style_axes(ax)
    save_plot("1")

def plot_grafico_2(df):
    share = pd.crosstab(
        df["Job_Title"],
        df["Risk_Category"],
        normalize="index"
    ).reindex(columns=["Low", "Medium", "High"], fill_value=0)

    dominant = share.idxmax(axis=1)
    share["_group"] = dominant.map({"Low": 0, "Medium": 1, "High": 2})
    share["_strength"] = share.max(axis=1)

    share = (
        share.sort_values(
            by=["_group", "_strength"],
            ascending=[True, False]
        )
        .drop(columns=["_group", "_strength"])
    )

    plt.figure(figsize=(5, 9))

    ax = sns.heatmap(
        share,
        annot=True,
        fmt=".0%",
        cmap="Blues",
        linewidths=1,
        cbar=False,
        annot_kws={"size": 12, "weight": "bold"}
    )

    ax.set_title(
        "Concentrazione del rischio per professione",
        fontsize=16,
        weight="bold",
        pad=20
    )

    ax.set_xlabel(
        "Fascia di rischio",
        fontsize=13,
        weight="bold"
    )

    ax.set_ylabel("")

    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=13)
    style_axes(ax)
    save_plot("2")

def plot_grafico_3(df):

    high_count = (df["Risk_Category"] == "High").sum()
    other_count = (df["Risk_Category"] != "High").sum()

    sizes = [high_count, other_count]

    total = sum(sizes)

    top_colors = [
        ACCENT,
        LIGHT_NEUTRAL
    ]

    desired_center_deg = 315
    high_theta = 360 * (sizes[0] / total) if total else 0
    startangle = (desired_center_deg + high_theta / 2) % 360

    fig, ax = plt.subplots(figsize=(9, 8))

    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=top_colors,
        startangle=startangle,
        counterclock=False,
        radius=1.0,
        center=(0, 0),
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.72,
        labeldistance=1.07,
        wedgeprops={
            "edgecolor": "none",
            "linewidth": 0
        }
    )

    outer_border = Circle(
        (0, 0),
        1.0,
        transform=ax.transData,
        fill=False,
        edgecolor=DARK,
        linewidth=1.4
    )

    ax.add_patch(outer_border)

    for t in autotexts:
        t.set_fontsize(17)
        t.set_fontweight("semibold")
        t.set_color("#1f1f1f")

    for i, t in enumerate(texts):
        t.set_fontsize(16)

        if i == 0:
            t.set_fontweight("semibold")
            t.set_color("#B00000")
        else:
            t.set_fontweight("regular")
            t.set_color("#5a5a5a")

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.30, 1.15)

    ax.axis("off")
    ax.set_aspect("equal")
    ax.legend(
        wedges,
        ["High Risk", "Medium + Low Risk"],
        loc="upper right",
        bbox_to_anchor=(1.15, 1.05),
        frameon=False,
        fontsize=13
    )
    save_plot("3")

def plot_grafico_4(df):
    highest_risk = df.loc[df["Automation_Probability_2030"].idxmax()]
    lowest_risk = df.loc[df["Automation_Probability_2030"].idxmin()]
    features = ["Tech_Growth_Factor","AI_Exposure_Index","Years_Experience","Average_Salary","Automation_Probability_2030"]
    labels = ["Tech Growth","AI Exposure","Experience","Salary","Automation Risk"]
    comparison = pd.DataFrame({f"Lowest Risk ({lowest_risk['Job_Title']})":lowest_risk[features],f"Highest Risk ({highest_risk['Job_Title']})":highest_risk[features]})
    for col in features:
        min_val = df[col].min()
        max_val = df[col].max()
        comparison.loc[col] = (
            comparison.loc[col] - min_val
        ) / (
            max_val - min_val
        )
    comparison.index = labels
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison.plot(kind="bar",ax=ax,color=[PRIMARY, ACCENT],width=0.75)
    ax.set_title("Confronto tra Professioni Estreme")
    ax.set_xlabel("")
    ax.set_ylabel("Valore Normalizzato")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y",linestyle=":",alpha=0.6)
    style_legend(ax)
    plt.xticks(rotation=20)
    style_axes(ax)
    save_plot("4")

def plot_grafico_5(df):
    tech_jobs = [
        "AI Engineer",
        "Data Scientist",
        "UX Researcher",
        "Software Engineer"
    ]

    df = df[df["Job_Title"] != "Research Scientist"].copy()

    df["Categoria"] = df["Job_Title"].apply(
        lambda x: "Tech" if x in tech_jobs else "Non-Tech"
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    tech = df.loc[
        df["Categoria"] == "Tech",
        "Automation_Probability_2030"
    ]

    nontech = df.loc[
        df["Categoria"] == "Non-Tech",
        "Automation_Probability_2030"
    ]

    bp = ax.boxplot(
        [tech, nontech],
        labels=["Tech", "Non-Tech"],
        notch=True,
        bootstrap=10000,
        patch_artist=True,
        widths=0.60,
        showfliers=False
    )

    colors = [
        PRIMARY,
        SECONDARY
    ]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.60)
        patch.set_edgecolor(DARK)
        patch.set_linewidth(1.2)

    for median in bp["medians"]:
        median.set_color(DARK)
        median.set_linewidth(2.2)

    for whisker in bp["whiskers"]:
        whisker.set_color(DARK)
        whisker.set_linewidth(1.2)

    for cap in bp["caps"]:
        cap.set_color(DARK)
        cap.set_linewidth(1.2)

    ax.set_xlabel("")

    ax.set_ylabel(
        "Probabilità di automazione",
        fontsize=12
    )

    ax.set_ylim(0, 1)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.4
    )

    sns.despine(
        top=True,
        right=True
    )
    style_axes(ax)
    save_plot("5")
    
def plot_grafico_6(df):
    bins = np.linspace(0, 1, 7)

    df = df.copy()
    df["Exposure_Bin"] = pd.cut(
        df["AI_Exposure_Index"],
        bins=bins,
        include_lowest=True
    )

    summary = (
        df.groupby("Exposure_Bin", observed=False)
          .agg(
              Exposure_Mean=("AI_Exposure_Index", "mean"),
              Risk_Mean=("Automation_Probability_2030", "mean")
          )
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        summary["Exposure_Mean"],
        summary["Risk_Mean"],
        marker="o",
        markersize=7,
        linewidth=2.5,
        color=SECONDARY,
        label="Rischio medio"
    )

    coef = np.polyfit(
        summary["Exposure_Mean"],
        summary["Risk_Mean"],
        1
    )

    x = np.linspace(0, 1, 100)
    y = coef[0] * x + coef[1]

    ax.plot(
        x,
        y,
        linestyle="--",
        linewidth=2,
        color=ACCENT,
        label="Trend lineare"
    )

    corr = df["AI_Exposure_Index"].corr(
        df["Automation_Probability_2030"]
    )

    ax.text(
        0.02,
        0.98,
        f"Correlazione: {corr:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.9,
            edgecolor="#cccccc"
        )
    )

    ax.set_title(
        "AI Exposure Index vs Probabilità di Automazione",
        fontsize=14,
        weight="bold"
    )

    ax.set_xlabel("AI Exposure Index")
    ax.set_ylabel("Probabilità di automazione")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    style_legend(ax, outside=False)
    style_axes(ax)
    save_plot("6")
    

def plot_grafico_7(df):
    jobs_to_show = ["Nurse", "Customer Support"]

    df_selected = df[df["Job_Title"].isin(jobs_to_show)]

    job_means = (
        df_selected
        .groupby("Job_Title", as_index=False)
        [["AI_Exposure_Index", "Automation_Probability_2030"]]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        "Nurse": POSITIVE,
        "Customer Support": ACCENT
    }

    for job in jobs_to_show:
        subset = df[df["Job_Title"] == job]

        ax.scatter(
            subset["AI_Exposure_Index"],
            subset["Automation_Probability_2030"],
            s=35,
            color=colors[job],
            alpha=0.18,
            edgecolors="none",
            zorder=1
        )

    for _, row in job_means.iterrows():
        job = row["Job_Title"]

        ax.vlines(
            row["AI_Exposure_Index"],
            0,
            row["Automation_Probability_2030"],
            color=colors[job],
            linestyle="--",
            linewidth=1.5,
            alpha=0.45,
            zorder=2
        )

        ax.scatter(
            row["AI_Exposure_Index"],
            row["Automation_Probability_2030"],
            s=320,
            color=colors[job],
            edgecolor=DARK,
            linewidth=1.3,
            zorder=5
        )

        ax.annotate(
            job,
            (
                row["AI_Exposure_Index"],
                row["Automation_Probability_2030"]
            ),
            xytext=(12, 8),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold"
        )

    ax.set_title(
        "Stessa esposizione all'AI, rischi di automazione diversi",
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    ax.set_xlabel("AI Exposure Index")
    ax.set_ylabel("Probabilità di automazione")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(alpha=0.3)

    sns.despine()
    style_axes(ax)
    save_plot("7")

def plot_grafico_8(df):
    df = df.copy()
    df["Risk_Category"] = pd.cut(
        df["Automation_Probability_2030"],
        bins=[0, 0.33, 0.66, 1],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    fig, ax = plt.subplots(figsize=(10, 7))

    low = df.loc[
        df["Risk_Category"] == "Low Risk",
        "Tech_Growth_Factor"
    ]

    medium = df.loc[
        df["Risk_Category"] == "Medium Risk",
        "Tech_Growth_Factor"
    ]

    high = df.loc[
        df["Risk_Category"] == "High Risk",
        "Tech_Growth_Factor"
    ]

    bp = ax.boxplot(
        [low, medium, high],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        notch=True,
        bootstrap=10000,
        patch_artist=True,
        widths=0.60,
        showfliers=False
    )

    colors = [
        PRIMARY,
        NEUTRAL,
        SECONDARY
    ]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.60)
        patch.set_edgecolor(DARK)
        patch.set_linewidth(1.2)

    for median in bp["medians"]:
        median.set_color(DARK)
        median.set_linewidth(2.2)

    for whisker in bp["whiskers"]:
        whisker.set_color(DARK)
        whisker.set_linewidth(1.2)

    for cap in bp["caps"]:
        cap.set_color(DARK)
        cap.set_linewidth(1.2)

    ax.set_title(
        "Tech Growth Factor nei diversi livelli di rischio",
        fontsize=14,
        weight="bold",
        pad=12
    )

    ax.set_xlabel(
        "Categoria di rischio di automazione"
    )

    ax.set_ylabel(
        "Tech Growth Factor"
    )

    ax.tick_params(
        axis="both",
        labelsize=11
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.4
    )

    sns.despine(
        top=True,
        right=True
    )
    style_axes(ax)
    save_plot("8")

def plot_grafico_9(df):
    grouped = (
        df.groupby("Years_Experience")["Automation_Probability_2030"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        grouped["Years_Experience"],
        grouped["Automation_Probability_2030"],
        marker="o",
        markersize=7,
        linewidth=2.5,
        color=PRIMARY,
        label="Rischio medio"
    )

    coef = np.polyfit(
        grouped["Years_Experience"],
        grouped["Automation_Probability_2030"],
        1
    )

    x = np.linspace(
        grouped["Years_Experience"].min(),
        grouped["Years_Experience"].max(),
        100
    )

    y = coef[0] * x + coef[1]

    ax.plot(
        x,
        y,
        linestyle="--",
        linewidth=2,
        color=ACCENT,
        label="Trend lineare"
    )

    corr = df["Years_Experience"].corr(
        df["Automation_Probability_2030"]
    )

    ax.text(
        0.02,
        0.98,
        f"Correlazione: {corr:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.9,
            edgecolor="#cccccc"
        )
    )

    ax.set_title("Andamento Rischio per Esperienza")
    ax.set_xlabel("Anni di Esperienza")
    ax.set_ylabel("Probabilità di Automazione")

    ax.set_ylim(0, 1)

    style_legend(ax, outside=False)
    style_axes(ax)
    save_plot("9")

def plot_grafico_10(df):
    edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
    data = [df[df['Education_Level'] == level]['Automation_Probability_2030'] for level in edu_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(data, labels=edu_order, patch_artist=True, notch=True)
    colors = [SECONDARY, PRIMARY, "#7A6FAC", POSITIVE]
    for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.set_title("Rischio per Titolo di Studio")
    ax.set_xlabel("Titolo di Studio"); ax.set_ylabel("Probabilità di Automazione")
    ax.set_ylim(0.0, 1.05); ax.grid(axis='y', linestyle=':', alpha=0.6)
    style_axes(ax)
    save_plot("10")

def plot_grafico_11(df):
    df = df.copy()

    df["Salary_Bin"] = (
        df["Average_Salary"] // 20000
    ) * 20000

    grouped = (
        df.groupby("Salary_Bin")["Automation_Probability_2030"]
        .mean()
        .reset_index()
    )

    labels = [
        f"{int(v/1000)}k-{int((v+20000)/1000)}k"
        for v in grouped["Salary_Bin"]
    ]

    x_idx = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        x_idx,
        grouped["Automation_Probability_2030"],
        marker="o",
        markersize=7,
        linewidth=2.5,
        color=SECONDARY,
        label="Rischio medio"
    )

    coef = np.polyfit(
        x_idx,
        grouped["Automation_Probability_2030"],
        1
    )

    x = np.linspace(
        x_idx.min(),
        x_idx.max(),
        100
    )

    y = coef[0] * x + coef[1]

    ax.plot(
        x,
        y,
        linestyle="--",
        linewidth=2,
        color=ACCENT,
        label="Trend lineare"
    )

    corr = df["Average_Salary"].corr(
        df["Automation_Probability_2030"]
    )

    ax.text(
        0.02,
        0.98,
        f"Correlazione: {corr:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.9,
            edgecolor="#cccccc"
        )
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels)

    ax.set_title("Rischio Medio per Fasce Salariali")
    ax.set_xlabel("Fascia Salariale Annuo")
    ax.set_ylabel("Probabilità di Automazione")

    ax.set_ylim(0, 1)

    style_legend(ax, outside=False)
    style_axes(ax)
    save_plot("11")

def plot_grafico_12(df):
    manuali = ['Retail Worker', 'Security Guard', 'Construction Worker', 'Truck Driver', 'Chef', 'Mechanic']
    cognitivi = ['Software Engineer', 'Data Scientist', 'Marketing Manager', 'UX Researcher', 'HR Specialist', 'Lawyer', 'Financial Analyst', 'Teacher', 'Research Scientist', 'AI Engineer']

    df_f = df[df['Job_Title'].isin(manuali + cognitivi)].copy()
    df_f['Type'] = df_f['Job_Title'].apply(lambda x: 'Manuale' if x in manuali else 'Cognitivo')

    d1 = df_f[df_f['Type'] == 'Manuale']['Automation_Probability_2030']
    d2 = df_f[df_f['Type'] == 'Cognitivo']['Automation_Probability_2030']

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot([d1, d2], labels=['Manuale', 'Cognitivo'], patch_artist=True, notch=True)
    colors = [SECONDARY, PRIMARY]

    for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color); patch.set_alpha(0.75)
    for median in box['medians']: median.set(color=ACCENT, linewidth=2.5)

    ax.set_title("Rischio per Tipologia di Mansione")
    ax.set_xlabel("Tipologia di Mansione"); ax.set_ylabel("Probabilità di Automazione")
    ax.set_ylim(0.0, 1.05); ax.grid(axis='y', linestyle=':', alpha=0.6)
    style_axes(ax)
    save_plot("12A")

def plot_grafico_13(df):
    manuali = ['Retail Worker', 'Security Guard', 'Construction Worker', 'Truck Driver', 'Chef', 'Mechanic']
    cognitivi = ['Software Engineer', 'Data Scientist', 'Marketing Manager', 'UX Researcher', 'HR Specialist', 'Lawyer', 'Financial Analyst', 'Teacher', 'Research Scientist', 'AI Engineer']
    all_means = df.groupby('Job_Title')['Automation_Probability_2030'].mean().sort_values()
    global_mean = df['Automation_Probability_2030'].mean()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = []
    for job in all_means.index:
        if job in manuali:
            colors.append(SECONDARY) # Arancione come grafico 11
        elif job in cognitivi:
            colors.append(PRIMARY) # Blu come grafico 11
        else:
            colors.append(NEUTRAL) # Grigio per i lavori ibridi/esclusi
    ax.barh(all_means.index, all_means.values, color=colors, height=0.6)
    ax.axvline(global_mean, color=NEUTRAL, linestyle='--', linewidth=1.5, label=f'Media Globale ({global_mean:.2f})')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=SECONDARY, lw=4, label='Mansione Manuale'),
        Line2D([0], [0], color=PRIMARY, lw=4, label='Mansione Cognitiva'),
        Line2D([0], [0], color=NEUTRAL, lw=4, label='Altre professioni'),
        Line2D([0], [0], color=NEUTRAL, lw=2, linestyle='--', label='Media Globale')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=11)

    ax.set_title("Distribuzione del Rischio per Tipologia di Mansione")
    ax.set_xlabel("Probabilità di Automazione")
    ax.set_ylabel("Professione")
    ax.set_xlim(0.0, 1.05)
    ax.grid(axis='x', linestyle=':', alpha=0.5)

    style_axes(ax)
    save_plot("13")

def plot_grafico_14(df):
    skill_cols = [f'Skill_{i}' for i in range(1, 11)]
    job_profiles = df.groupby('Job_Title')[['Automation_Probability_2030'] + skill_cols].mean().sort_values(by='Automation_Probability_2030', ascending=False)
    matrix = job_profiles[skill_cols].values
    labels = [f"{j} ({r:.2f})" for j, r in zip(job_profiles.index, job_profiles['Automation_Probability_2030'])]

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0.45, vmax=0.58)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color=DARK, fontsize=10, fontweight='bold')

    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(10)); ax.set_xticklabels(skill_cols)

    cbar = fig.colorbar(im, ax=ax, label='Intensità Skill')

    ax.set_title("Mappatura delle competenze per suscettibilità al rischio", pad=20, fontsize=16)
    ax.set_xlabel("Dimensioni Competenze"); ax.set_ylabel("Professioni (Rischio decrescente)")
    style_axes(ax)
    save_plot("14")
    
def plot_grafico_15(df):
    skill_cols = [f"Skill_{i}" for i in range(1, 11)]
    highest_risk = df.loc[df["Automation_Probability_2030"].idxmax()]
    lowest_risk = df.loc[df["Automation_Probability_2030"].idxmin()]
    x = np.arange(len(skill_cols))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2,lowest_risk[skill_cols],width,color=PRIMARY,label=f"Lowest Risk ({lowest_risk['Job_Title']})")
    ax.bar(x + width / 2,highest_risk[skill_cols],width,color=ACCENT,label=f"Highest Risk ({highest_risk['Job_Title']})")
    ax.set_xticks(x)
    ax.set_xticklabels(skill_cols)
    ax.set_xlabel("Skills")
    ax.set_ylabel("Valore della Skill")
    ax.set_title("Profilo delle Competenze: Lavori a Rischio Minimo vs Massimo")
    ax.grid(axis="y",linestyle=":",alpha=0.6)
    style_legend(ax)
    style_axes(ax)
    save_plot("15")

def plot_grafico_16(df):
    skill_cols = [f'Skill_{i}' for i in range(1, 11)]
    avg = df.groupby('Risk_Category')[skill_cols].mean().reindex(['Low', 'Medium', 'High'])

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap('Blues')
    colors = cmap(np.linspace(0.3, 0.9, 10))

    left = np.zeros(len(avg))

    for i, skill in enumerate(skill_cols):
        ax.barh(avg.index, avg[skill], left=left, color=colors[i], label=skill, edgecolor='white', height=0.6)
        left += avg[skill].values

    ax.set_title("Composizione delle Skill per Categoria di Rischio", pad=20, fontsize=14)
    ax.set_xlabel("Somma Intensità Skill (Valore Cumulativo)")
    ax.set_ylabel("Categoria di Rischio")
    ax.set_xticks(np.arange(0, 5.5, 0.5))
    ax.legend(title="Competenze", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=11)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    style_axes(ax)
    save_plot("16")

def plot_grafico_17(df):
    skill_cols = [f"Skill_{i}" for i in range(1, 11)]
    df = df.dropna(subset=["Risk_Category", "Automation_Probability_2030"])
    df_high = df[df["Risk_Category"] == "High"]
    df_low = df[df["Risk_Category"] == "Low"]
    corr_high = df_high[skill_cols + ["Automation_Probability_2030"]].corr()
    corr_low = df_low[skill_cols + ["Automation_Probability_2030"]].corr()
    risk_corr_high = corr_high["Automation_Probability_2030"].drop("Automation_Probability_2030")
    risk_corr_low = corr_low["Automation_Probability_2030"].drop("Automation_Probability_2030")
    delta = (risk_corr_high - risk_corr_low).sort_values()
    colors = [ACCENT if v > 0 else PRIMARY for v in delta.values]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(delta.index,delta.values,color=colors)
    ax.axvline(0, color=DARK, linewidth=1)
    ax.set_xlim(-1.0, 1.0)
    ax.set_title("Analisi Delta: Impatto delle Skill nei Lavori ad Alto vs Basso Rischio")
    ax.set_xlabel("Δ Correlazione (Alto Rischio − Basso Rischio)")
    ax.set_ylabel("Skills")
    ax.grid(axis="x",linestyle=":",alpha=0.6)
    legend_patches = [
        mpatches.Patch(
            color=ACCENT,
            label="Maggiore correlazione nei lavori ad alto rischio"
        ),
        mpatches.Patch(
            color=PRIMARY,
            label="Maggiore correlazione nei lavori a basso rischio"
        )
    ]

    ax.legend(
        handles=legend_patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    style_axes(ax)
    save_plot("17")
df = load_data()
for f in [
    plot_grafico_1,
    plot_grafico_2,
    plot_grafico_3,
    plot_grafico_4,
    plot_grafico_5,
    plot_grafico_6,
    plot_grafico_7,
    plot_grafico_8,
    plot_grafico_9,
    plot_grafico_10,
    plot_grafico_11,
    plot_grafico_12,
    plot_grafico_13,
    plot_grafico_14,
    plot_grafico_15,
    plot_grafico_16,
    plot_grafico_17
]:
    f(df)
print(f"Grafici salvati in: {output_dir}")
