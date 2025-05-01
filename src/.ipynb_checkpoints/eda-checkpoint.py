import pandas as pd
import matplotlib.pyplot as plt

def load_and_explore_data(fake_path, true_path):
    fake_data = pd.read_csv(fake_path)
    true_data = pd.read_csv(true_path)

    fake_data['label'] = 'fake'
    true_data['label'] = 'true'

    combined_data = pd.concat([fake_data, true_data], ignore_index=True)

    # Affiche des informations de base
    print("Fake news dataset:")
    print(fake_data.info())
    print("\nTrue news dataset:")
    print(true_data.info())
    print("\nCombined dataset:")
    print(combined_data['label'].value_counts())

    # Visualiser la répartition des classes
    combined_data['label'].value_counts().plot(kind='bar', title='Distribution of Fake and True News')
    plt.show()

    return combined_data