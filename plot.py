import matplotlib.pyplot as plt
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_data_from_sheet(sheet_name):
    URL = f"https://sheets.googleapis.com/v4/spreadsheets/1pFru8G5YRUxsxVn8gXKZX08VGgbI_SMqtMJBzAskqSY/values/{sheet_name}!A1:Z?alt=json&key={key}"
    try:
        response = requests.get(URL)
        response.raise_for_status()

        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the data from {sheet_name}: {e}")
        return None

def plot_number(sheet, readings):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    data = readings['values']
    num_of_threads_array = data[4]
    n = len(data)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    title_set = {"LoRA","LoRA+SVD","Fourier","Frame (3)"}
    colors = {"LoRA":"#a93226","LoRA+SVD":"#186a3b","Fourier": "#005f73","Frame (3)":"#a569bd"}

    for i in range(5, n):
        title = data[i][0]
        if title not in title_set:
            continue

        x_labels = data[i][1:]
        y_labels = num_of_threads_array[1:]

        if not x_labels or x_labels[0] == '-':
            continue

        x_values = list(map(float, x_labels))
        y_values = list(map(str, y_labels))

        plt.plot(
            y_values, x_values,
            marker='^',
            color=colors[title],
            markerfacecolor='white',
            markeredgewidth=2,
            markeredgecolor=colors[title],
            markersize=10,
            linewidth=3,
            label=title  # Use "LoRA" or "LoRA+SVD"
        )

        for x, y in zip(y_values, x_values):
            plt.annotate(
                f"({x}, {y:.2f})",
                xy=(x, y),
                xytext=(3, 3),
                textcoords='offset points',
                fontsize=12,
                ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[title], lw=1.5),
                arrowprops=dict(arrowstyle='->', color=colors[title])
            )

    # Final plot settings
    plt.legend(loc='best', fontsize=13, frameon=True)
    plt.xlabel('Number of tokens', fontsize=15, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=15, fontweight='bold')
    plt.title("Num input tokens vs. Execution Time (ms)", fontsize=17, fontweight='bold', color='#333333')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, color='lightgray', linestyle='--', linewidth=1)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    plt.tight_layout()

    output_dir = sheet
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "Fourier+Hadamard+Frame+stuff.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.clf()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_model(sheet, readings):
    data = readings['values']
    query_model_names = data[18][1:]
    query_data = []
    print(query_model_names)
    title_map = {
        "LoRA": "LoRA",
        "LoRA+SVD": "LoRA+SVD",
        "Hadamard (Type 1)": "Hadamard\n(Type 1)",
        "Fourier (Type 2)": "Fourier\n(Type 2)",
        "Frame (Type 3)": "Frame\n(Type 3)"
    }

    for i in range(19, 30):
        if data[i]:
            title = data[i][0]
            if title in title_map:
                y_values = list(map(float, data[i][1:]))
                for model, value in zip(query_model_names, y_values):
                    if model == "Gemma-2-9B":
                        query_data.append({'Model': model, 'Metric': title_map[title], 'Value': value})

    df_query = pd.DataFrame(query_data)
    metric_order = ["LoRA", "LoRA+SVD", "Hadamard\n(Type 1)", "Fourier\n(Type 2)", "Frame\n(Type 3)"]
    df_query['Metric'] = pd.Categorical(df_query['Metric'], categories=metric_order, ordered=True)
    df_query.sort_values('Metric', inplace=True)

    # ===== Simplified Plot Using plt =====
    custom_colors = ["#005f73", "#0a9396", "#4b1d91", "#ae2012", "#ee9b00"]
    x_labels = df_query['Metric'].tolist()
    y_values = df_query['Value'].tolist()

    bar_width = 0.2
    x = np.arange(len(x_labels))

    plt.figure(figsize=(7, 7))
    bars = plt.bar(x, y_values, width=0.7, color=custom_colors)
    plt.margins(x=0.2)

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f'{yval:.1f}',
         ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xticks(x, x_labels, fontsize=12)
    plt.title(f"{data[17][0]} for Gemma-2-9B", fontsize=18, fontweight='bold')
    plt.xlabel("Method", fontsize=12, fontweight='bold')
    plt.ylabel("Execution Time (ms)", fontsize=12, fontweight='bold')
    plt.grid(True, axis='y', linestyle='--')

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    # Save plot
    output_dir = sheet
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{data[3][0]}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()



    # value_model_names = data[18][1:]
    # value_data = []

    # for i in range(19, 30):
    #     title = data[i][0]
    #     y_labels = data[i][1:]
    #     if y_labels[0] == '-':
    #         continue
    #     y_values = list(map(float, y_labels))
    #     for model, value in zip(value_model_names, y_values):
    #         value_data.append({'Model': model, 'Metric': title, 'Value': value})

    # df_value = pd.DataFrame(value_data)

    # plt.figure(figsize=(12, 7))
    # sns.lineplot(data=df_value, x='Model', y='Value', hue='Metric', marker='o', linewidth=2)
    # plt.title(data[17][0], fontsize=18, fontweight='bold')
    # plt.xlabel("Models", fontsize=12)
    # plt.ylabel("Values", fontsize=12)
    # plt.xticks(rotation=45)
    # plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.grid(True)
    # output_dir = sheet
    # os.makedirs(output_dir, exist_ok=True)
    # plot_path = os.path.join(output_dir, f"{data[17][0]}.png")
    # plt.savefig(plot_path, dpi=300)

def main():
    num_threads_per_block_data = get_data_from_sheet("Num_threads_per_block")
    num_of_tokens_data = get_data_from_sheet("Number of tokens")
    block_size_data = get_data_from_sheet("Block size")
    num_of_coeffs_in_C_data = get_data_from_sheet("num of coeffs in C")
    model_data = get_data_from_sheet("Model")

    plot_number("Number of tokens", num_of_tokens_data)
    # plot_number("Number of tokens", num_of_tokens_data)
    # plot_number("Block size", block_size_data)
    # plot_number("Num_threads_per_block", num_threads_per_block_data)
    # plot_model("Model", model_data)

if __name__ == '__main__':
    main()