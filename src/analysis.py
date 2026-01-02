"""Analysis and visualization of demographic estimation results."""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_age_accuracy(results: list[dict], output_path: str) -> None:
    """Plot age estimation accuracy with enhanced metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    actual_ages = []
    estimated_ages = []
    confidences = []
    errors = []

    for result in results:
        print(type(result))
        actual = result["actual"]
        estimated = result["estimated"]

        actual_mid = np.mean([float(x) for x in actual["age_range"].split("-")])
        print(estimated)
        est_mid = np.mean([estimated["age_lower"], estimated["age_upper"]])

        actual_ages.append(actual_mid)
        estimated_ages.append(est_mid)
        confidences.append(estimated["age_confidence"])
        errors.append(abs(actual_mid - est_mid))

    # Accuracy scatter plot
    scatter = ax1.scatter(
        actual_ages,
        estimated_ages,
        c=confidences,
        s=150,
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
    )
    ax1.plot([0, 70], [0, 70], "r--", linewidth=2, label="Perfect")
    ax1.set_xlabel("Actual Age (midpoint)", fontsize=12)
    ax1.set_ylabel("Estimated Age (midpoint)", fontsize=12)
    ax1.set_title("Age Estimation Accuracy", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="Confidence")

    # Confidence distribution
    ax2.hist(confidences, bins=10, edgecolor="black", color="skyblue", alpha=0.7)
    ax2.axvline(
        np.mean(confidences),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(confidences):.2f}",
    )
    ax2.set_xlabel("Confidence Level", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Age Confidence Distribution", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Error distribution
    ax3.hist(errors, bins=10, edgecolor="black", color="coral", alpha=0.7)
    ax3.axvline(
        np.mean(errors),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean Error: {np.mean(errors):.1f}y",
    )
    ax3.set_xlabel("Absolute Error (years)", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)
    ax3.set_title("Age Error Distribution", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Error vs Confidence
    ax4.scatter(
        confidences, errors, s=150, alpha=0.7, edgecolors="black", color="green"
    )
    ax4.set_xlabel("Confidence Level", fontsize=12)
    ax4.set_ylabel("Absolute Error (years)", fontsize=12)
    ax4.set_title("Error vs Confidence", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_income_accuracy(results: list[dict], output_path: str) -> None:
    """Plot income estimation accuracy with error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    personas = []
    actual_incomes = []
    estimated_incomes = []
    confidences = []

    for i, result in enumerate(results):
        actual = result["actual"]
        estimated = result["estimated"]

        actual_range = actual["income_range"].replace("$", "").replace(",", "")
        actual_mid = np.mean([float(x) for x in actual_range.split("-")])

        est_mid = np.mean([estimated["income_lower"], estimated["income_upper"]])

        personas.append(f"P{i + 1}")
        actual_incomes.append(actual_mid / 1000)
        estimated_incomes.append(est_mid / 1000)
        confidences.append(estimated["income_confidence"])

    x = np.arange(len(personas))
    width = 0.35

    # Bar chart comparison
    ax1.bar(
        x - width / 2,
        actual_incomes,
        width,
        label="Actual",
        color="steelblue",
        alpha=0.8,
    )
    ax1.bar(
        x + width / 2,
        estimated_incomes,
        width,
        label="Estimated",
        color="coral",
        alpha=0.8,
    )

    ax1.set_xlabel("Persona", fontsize=12)
    ax1.set_ylabel("Income ($1000s)", fontsize=12)
    ax1.set_title("Income Estimation Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(personas)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    # Scatter plot with confidence
    scatter = ax2.scatter(
        actual_incomes,
        estimated_incomes,
        c=confidences,
        s=150,
        cmap="plasma",
        alpha=0.7,
        edgecolors="black",
    )
    max_income = max(max(actual_incomes), max(estimated_incomes))
    ax2.plot([0, max_income], [0, max_income], "r--", linewidth=2, label="Perfect")
    ax2.set_xlabel("Actual Income ($1000s)", fontsize=12)
    ax2.set_ylabel("Estimated Income ($1000s)", fontsize=12)
    ax2.set_title("Income Accuracy Scatter", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label="Confidence")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_categorical_accuracy(results: list[dict], output_path: str) -> None:
    """Plot accuracy for categorical variables with stats."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    education_correct = 0
    sex_correct = 0
    visa_correct = 0

    for result in results:
        actual = result["actual"]
        estimated = result["estimated"]

        if actual["education"] == estimated["education"]:
            education_correct += 1
        if actual["sex"] == estimated["sex"]:
            sex_correct += 1
        if actual["visa_status"] == estimated["visa_status"]:
            visa_correct += 1

    total = len(results)

    # Pie charts
    colors = ["#90EE90", "#FFB6C6"]
    for ax, correct, title in [
        (ax1, education_correct, "Education"),
        (ax2, sex_correct, "Sex"),
        (ax3, visa_correct, "Visa Status"),
    ]:
        values = [correct, total - correct]
        wedges, texts, autotexts = ax.pie(
            values,
            labels=["Correct", "Incorrect"],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 11, "weight": "bold"},
        )
        ax.set_title(f"{title} Accuracy", fontsize=14, fontweight="bold", pad=20)

    # Summary bar chart
    categories = ["Education", "Sex", "Visa Status"]
    accuracies = [
        (education_correct / total) * 100,
        (sex_correct / total) * 100,
        (visa_correct / total) * 100,
    ]

    bars = ax4.bar(
        categories,
        accuracies,
        color=["steelblue", "coral", "lightgreen"],
        alpha=0.8,
        edgecolor="black",
    )
    ax4.set_ylabel("Accuracy (%)", fontsize=12)
    ax4.set_title("Overall Categorical Accuracy", fontsize=14, fontweight="bold")
    ax4.set_ylim(0, 110)
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_report(results: list[dict]) -> str:
    """Create enhanced text summary of results."""
    total = len(results)
    report = [f"Analysis Summary ({total} personas)\n"]
    report.append("=" * 50)

    age_errors = []
    income_errors = []
    education_correct = 0
    sex_correct = 0
    visa_correct = 0

    for result in results:
        actual = result["actual"]
        estimated = result["estimated"]

        # Age metrics
        actual_age = np.mean([float(x) for x in actual["age_range"].split("-")])
        est_age = np.mean([estimated["age_lower"], estimated["age_upper"]])
        age_errors.append(abs(actual_age - est_age))

        # Income metrics
        actual_income = actual["income_range"].replace("$", "").replace(",", "")
        actual_income_mid = np.mean([float(x) for x in actual_income.split("-")])
        est_income = np.mean([estimated["income_lower"], estimated["income_upper"]])
        income_errors.append(
            abs(actual_income_mid - est_income) / actual_income_mid * 100
        )

        # Categorical metrics
        if actual["education"] == estimated["education"]:
            education_correct += 1
        if actual["sex"] == estimated["sex"]:
            sex_correct += 1
        if actual["visa_status"] == estimated["visa_status"]:
            visa_correct += 1

    report.append("\nNumerical Demographics:")
    report.append(f"  Average age error: {np.mean(age_errors):.1f} years")
    report.append(f"  Max age error: {np.max(age_errors):.1f} years")
    report.append(f"  Average income error: {np.mean(income_errors):.1f}%")

    report.append("\nCategorical Demographics:")
    report.append(
        f"  Education accuracy: {education_correct}/{total} "
        f"({education_correct / total * 100:.1f}%)"
    )
    report.append(
        f"  Sex accuracy: {sex_correct}/{total} ({sex_correct / total * 100:.1f}%)"
    )
    report.append(
        f"  Visa status accuracy: {visa_correct}/{total} "
        f"({visa_correct / total * 100:.1f}%)"
    )

    return "\n".join(report)
