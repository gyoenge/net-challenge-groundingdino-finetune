import pandas as pd
import matplotlib.pyplot as plt
import os

def extract_title_from_filename(file_name):
    # extract string after "tag-" 
    title = file_name.split("tag-")[-1].split(" ")[0]
    # replace first '_' to '/' 
    title = title.replace('_', '/', 1)
    return title

def plot_graph_from_csv(file_path):
    # read CSV file 
    data = pd.read_csv(file_path)
    
    # extract graph title from filename
    title = extract_title_from_filename(os.path.basename(file_path))
    
    # plot 
    plt.figure(figsize=(10, 6))
    plt.plot(data["Step"], data["Value"], label=title, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(left=0.5, right=0.9, bottom=0.5, top=0.9)
    plt.show()


def plot_graphs_from_directory(directory_path):
    # plot for all csv files in selected directory 
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            plot_graph_from_csv(os.path.join(directory_path, file_name))

# run
directory_path = "epochs_graph_data/"
plot_graphs_from_directory(directory_path)

