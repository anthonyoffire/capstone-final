from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def print_ascii_confusion_matrix(
    age_threshold,
    probability_threshold,
    tn,
    fp,
    fn,
    tp,
    run_label,
):
    label_width = 23
    cell_width = 19
    actual_under = f"Actual < {age_threshold}"
    actual_over = f"Actual >= {age_threshold}"
    predicted_under = f"Predicted < {age_threshold}"
    predicted_over = f"Predicted >= {age_threshold}"
    border = (
        f"+{'-' * (label_width + 2)}"
        f"+{'-' * (cell_width + 2)}"
        f"+{'-' * (cell_width + 2)}+"
    )

    print(
        f"\nConfusion matrix ({run_label}, age cutoff {age_threshold}, "
        f"probability cutoff {probability_threshold:g})"
    )
    print(border)
    print(
        f"| {'Predicted / Actual':<{label_width}} "
        f"| {actual_under:>{cell_width}} | {actual_over:>{cell_width}} |"
    )
    print(border)
    print(
        f"| {predicted_under:<{label_width}} "
        f"| {f'TN {tn:,}':>{cell_width}} | {f'FN {fn:,}':>{cell_width}} |"
    )
    print(
        f"| {predicted_over:<{label_width}} "
        f"| {f'FP {fp:,}':>{cell_width}} | {f'TP {tp:,}':>{cell_width}} |"
    )
    print(border)


def save_confusion_matrix(
    output_dir,
    age_threshold,
    tn,
    fp,
    fn,
    tp,
    accuracy,
    recall,
    specificity,
):
    actual_totals = (tn + fp, fn + tp)
    predicted_totals = (tn + fn, fp + tp)
    total = sum(actual_totals)
    matrix = (
        (tn, fn, predicted_totals[0]),
        (fp, tp, predicted_totals[1]),
        (actual_totals[0], actual_totals[1], total),
    )
    colors = (
        ("#82e98f", "#ff8b8b", "#e9edf2"),
        ("#ff8b8b", "#82e98f", "#e9edf2"),
        ("#e9edf2", "#e9edf2", "#d9e1ea"),
    )
    column_widths = (1.7, 1.8, 1.8, 1.7)
    x_edges = np.cumsum((0, *column_widths))
    table_center_x = 0.5 * (x_edges[0] + x_edges[-1])
    actual_classes_center_x = 0.5 * (x_edges[1] + x_edges[3])

    fig, ax = plt.subplots(figsize=(9, 6.2), layout="constrained")
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#f7f8fa")
    ax.set_xlim(x_edges[0] - 0.35, x_edges[-1])
    ax.set_ylim(-0.55, 5)
    ax.axis("off")

    ax.text(
        table_center_x,
        4.82,
        "CONFUSION MATRIX",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#243447",
    )
    ax.text(
        table_center_x,
        4.48,
        f"Accuracy {accuracy:.1%}   |   Recall {recall:.1%}   |   "
        f"Specificity {specificity:.1%}",
        ha="center",
        va="center",
        fontsize=11,
        color="#526579",
    )
    ax.text(
        actual_classes_center_x,
        4.13,
        "ACTUAL",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#526579",
    )

    header_y = 3
    header_labels = (
        "",
        f"Actual <{age_threshold}",
        f"Actual >={age_threshold}",
        "SUM",
    )
    header_colors = ("#d9e1ea", "#d9e1ea", "#d9e1ea", "#d9e1ea")
    for column, text in enumerate(header_labels):
        x0, x1 = x_edges[column], x_edges[column + 1]
        ax.add_patch(plt.Rectangle(
            (x0, header_y),
            x1 - x0,
            1,
            facecolor=header_colors[column],
            edgecolor="white",
            linewidth=2,
        ))
        ax.text(
            (x0 + x1) / 2,
            header_y + 0.5,
            text,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#243447",
        )

    row_labels = (
        f"Pred <{age_threshold}",
        f"Pred >={age_threshold}",
        "SUM",
    )
    for matrix_row in range(3):
        y0 = 2 - matrix_row
        label_x0, label_x1 = x_edges[0], x_edges[1]
        ax.add_patch(plt.Rectangle(
            (label_x0, y0),
            label_x1 - label_x0,
            1,
            facecolor="#d9e1ea",
            edgecolor="white",
            linewidth=2,
        ))
        ax.text(
            (label_x0 + label_x1) / 2,
            y0 + 0.5,
            row_labels[matrix_row],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#243447",
        )

        for matrix_column in range(3):
            x0, x1 = x_edges[matrix_column + 1], x_edges[matrix_column + 2]
            count = matrix[matrix_row][matrix_column]
            if matrix_row < 2 and matrix_column < 2:
                actual_count = actual_totals[matrix_column]
                class_percentage = (
                    f"{count / actual_count:.1%} of class"
                    if actual_count
                    else "n/a"
                )
                annotation = (
                    f"{count:,}\n"
                    f"{count / total:.2%} of total\n"
                    f"{class_percentage}"
                )
            else:
                annotation = f"{count:,}"

            ax.add_patch(plt.Rectangle(
                (x0, y0),
                x1 - x0,
                1,
                facecolor=colors[matrix_row][matrix_column],
                edgecolor="white",
                linewidth=2,
            ))
            ax.text(
                (x0 + x1) / 2,
                y0 + 0.5,
                annotation,
                ha="center",
                va="center",
                fontsize=10 if matrix_row < 2 and matrix_column < 2 else 11,
                fontweight="bold",
                color="#203040",
                linespacing=1.35,
            )

    ax.text(
        x_edges[0] - 0.12,
        2,
        "PREDICTED",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
        fontweight="bold",
        color="#526579",
    )
    matrix_path = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(
        matrix_path,
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return matrix_path
