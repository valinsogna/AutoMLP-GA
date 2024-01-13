import matplotlib.pyplot as plt
import numpy as np

def read_file(file_path):
    """Read the file and extract accuracy values."""
    accuracies = []
    with open(file_path, 'r') as file:
        print(file_path)
        for line in file:
            if line.startswith('0') or line.startswith('1'):
                try:
                    accuracy = float(line.strip())
                    accuracies.append(accuracy)
                except ValueError:
                    pass  # Ignore lines that are not numbers
    return accuracies

def compute_statistics(values):
    """Compute mean, median, and standard deviation."""
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)
    return mean, median, std

def plot_distribution(values, file_path):
    """Plot the distribution of accuracy values."""
    plt.hist(values, bins=10, alpha=0.7, color='blue')
    plt.title('Distribution of Accuracy Values')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.show()
    #Save plot in file_path - last string "network_architectures.txt" and name it "final_accuracy_distribution.png":
    plt.savefig(dir_path + 'final_accuracy_distribution.png')


def main(file_path):
    accuracies = read_file(file_path)
    dir_path=file_path[:-25]
    mean, median, std = compute_statistics(accuracies)
    #Save mean, median and std in file txt named "final_accuracy_statistics.txt":
    with open(dir_path + 'final_accuracy_statistics.txt', 'w') as file:
        file.write(f'Mean: {mean:.4f}, Median: {median:.4f}, Standard Deviation: {std:.4f}')
    print(f'Mean: {mean:.4f}, Median: {median:.4f}, Standard Deviation: {std:.4f}')

    plot_distribution(accuracies, dir_path)

if __name__ == "__main__":
    file_path = '/Users/valeriainsogna/Desktop/AutoMLP-GA/results/run_20240113-012449/network_architectures.txt'  # Replace with your file path
    main(file_path)
