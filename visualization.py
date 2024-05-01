import numpy as np
import matplotlib.pyplot as plt

def plot_all_errors(*errors):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Wykres pudełkowy
    bp = ax.boxplot(errors, patch_artist=True)

    # Etykiety osi X
    ax.set_xticklabels(['Model 2016-2017', 'Model 2016-2018', 'Model 2016-2019', 'Model 2016-2020', 'Model 2016-2021', 'Model 2016-2022'], fontsize=12)

    # Dodanie tytułu
    plt.title("Errors for all models", fontsize=16)

    # Personalizacja pudełek
    colors = plt.cm.Set3(np.linspace(0, 1, len(errors)))
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)

    # Grubsza kreska mediany
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    # Dodanie poziomych linii siatki
    ax.yaxis.grid(True)
    ax.tick_params(axis='both', which='both', labelsize=10)

    # Usunięcie górnych i prawych osi
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Pokazanie wykresu
    plt.show()

    return None

def plot_feature_importances(variable_names, median_importances):
    sorted_importances = sorted(zip(variable_names, median_importances), key=lambda x: x[1], reverse=False)
    top_importances = sorted_importances[-25:]
    top_variable_names, top_median_importances = zip(*top_importances)

    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_variable_names)))[::-1]
    plt.barh(top_variable_names, top_median_importances, color=colors)
    plt.title('Feature Importances', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'feature_importance_plot.svg', format='svg')
    plt.show()

    return None