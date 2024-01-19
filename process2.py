import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def read_file(file_path):
    """Read the file and extract network specifications along with accuracy values."""
    networks = []
    with open(file_path, 'r') as file:
        spec = None
        for line in file:
            if line.startswith('{'):
                spec = line.strip()  # Store the spec line
            elif line.strip().replace('.', '', 1).isdigit():
                # This line should be the accuracy if it's a digit (with a possible decimal point)
                accuracy = float(line.strip())
                if spec:
                    networks.append((spec, accuracy))
                    spec = None  # Reset spec for the next entry
            # The '---' line is ignored
    return networks



def compute_statistics(values):
    """Compute mean, median, and standard deviation."""
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)
    return mean, median, std

def plot_distribution(values, dir_path):
    """Plot the distribution of accuracy values."""
    plt.hist(values, bins=10, alpha=0.7, color='blue')
    plt.title('Distribution of Accuracy Values')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.show()
    #Save plot in file_path - last string "network_architectures.txt" and name it "final_accuracy_distribution.png":
    plt.savefig(dir_path + 'final_accuracy_distribution.png')


def main(file_path):
    #accuracies = read_file(file_path)
    dir_path=file_path[:-25]
    #mean, median, std = compute_statistics(accuracies)
    #Save mean, median and std in file txt named "final_accuracy_statistics.txt":
    # with open(dir_path + '/final_accuracy_statistics.txt', 'w') as file:
    #     file.write(f'Mean: {mean:.4f}, Median: {median:.4f}, Standard Deviation: {std:.4f}')
    # print(f'Mean: {mean:.4f}, Median: {median:.4f}, Standard Deviation: {std:.4f}')

    #plot_distribution(accuracies, dir_path)

    #Print top 10% networks:
    # accuracies.sort(reverse=True)
    # top_10 = int(len(accuracies)/10)
    # print(f"Printing top 10% networks:")
    # for i in range(top_10):
    #     print(f"{i+1}. {accuracies[i]:.4f}")
    # # Save top 10% networks in file txt named "top_10_networks.txt" along with their rank and network spec:
    # with open(dir_path + '/top_10_networks.txt', 'w') as file:
    #     for i in range(top_10):
    #         file.write(f"{i+1}. {accuracies[i]:.4f}\n") 
    # print(f"Saved top 10% networks in {dir_path}/top_10_networks.txt")
    networks = read_file(file_path)
    # ... other parts of the function ...

    networks.sort(key=lambda x: x[1], reverse=True)
    top_10_percent_index = int(len(networks) * 0.1)
    top_10_percent_networks = networks[:top_10_percent_index]

    with open(dir_path + '/top_10_networks.txt', 'w') as file:
        for i, (spec, acc) in enumerate(top_10_percent_networks, start=1):
            file.write(f"{i}. {spec} - Accuracy: {acc:.4f}\n")
    print(f"Saved top 10% networks in {dir_path}/top_10_networks.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate networks.')
    parser.add_argument('--file', type=str, required=True, help='Path to file containing accuracies.')
    args = parser.parse_args()
    file_path = args.file
    main(file_path)
