import matplotlib.pyplot as plt
import requests
import os

def get_data_from_sheet(sheet_name):
    URL = f"https://sheets.googleapis.com/v4/spreadsheets/1pFru8G5YRUxsxVn8gXKZX08VGgbI_SMqtMJBzAskqSY/values/{sheet_name}!A1:Z?alt=json&key=AIzaSyBo96VKoz6GWWaKuBZ0rMeTM8lF3i_Mx2w"
    try:
        response = requests.get(URL)
        response.raise_for_status()

        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the data from {sheet_name}: {e}")
        return None

def plot_number_per_block(sheet, readings):
    data = readings['values']
    num_of_threads_array = data[4]
    n = len(data)
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Optional: set Seaborn theme
    sns.set_theme(style="whitegrid")

    for i in range(5, n):
        plt.figure(figsize=(12, 8))

        x_labels = data[i][1:]
        y_labels = num_of_threads_array[1:]

        x_values = list(map(float, x_labels))
        y_values = list(map(int, y_labels))

        # Use a more distinct line and marker
        plt.plot(y_values, x_values, linestyle='dashed', marker='X', color='#00796b', markersize=10, linewidth=3)

        # Enhance axis labels and title
        plt.xlabel('Number of Threads', fontsize=15, fontweight='bold')
        plt.ylabel('Execution Time (ms)', fontsize=15, fontweight='bold')
        plt.title(data[i][0], fontsize=16, fontweight='bold')

        # Annotate each point
        for x, y in zip(y_values, x_values):
            plt.annotate(f"({x}, {y:.2f})", xy=(x, y), xytext=(40, 10), textcoords='offset points',
                        fontsize=13, color='black', ha='center')

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, color='gray', linestyle='--', linewidth=1)
        plt.tight_layout()

        # Save the figure
        output_dir = sheet
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{data[i][0]}.png")
        plt.savefig(plot_path, dpi=300)
        plt.clf()
        break



def main():
    num_threads_per_block_data = get_data_from_sheet("Num_threads_per_block")
    num_of_tokens_data = get_data_from_sheet("Number of tokens")
    block_size_data = get_data_from_sheet("Block size")
    num_of_coeffs_in_C_data = get_data_from_sheet("num of coeffs in C")
    model_data = get_data_from_sheet("Model")

    plot_number_per_block("Num_threads_per_block",num_threads_per_block_data)
    plot_number_per_block("Number of tokens", num_of_tokens_data)

if __name__ == '__main__':
    main()