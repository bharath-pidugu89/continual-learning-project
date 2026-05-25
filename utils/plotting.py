import matplotlib.pyplot as plt


def plot_results(methods, accuracies, title, save_path):

    plt.figure(figsize=(8,5))

    plt.bar(methods, accuracies)

    plt.ylabel("Accuracy (%)")

    plt.title(title)

    plt.ylim(0, 100)

    plt.savefig(save_path)

    plt.show()