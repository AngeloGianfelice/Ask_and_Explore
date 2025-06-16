import numpy as np
import torch
import matplotlib.pyplot as plt
import random # For set_seed

# --- Utility Functions ---
def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_metric(data, figure_file, xlabel, ylabel, title):
    """Plots the learning curve."""
    running_avg = np.zeros(len(data))
    running_std = np.zeros(len(data))
    for i in range(len(data)):
        running_avg[i] = np.mean(data[max(0, i-100):(i+1)])
        running_std[i] = np.std(data[max(0, i-100):(i+1)])

    x = [i+1 for i in range(len(data))]
    plt.plot(x, running_avg)
    # Plot standard deviation as a light-colored stripe
    plt.fill_between(x, running_avg - running_std, running_avg + running_std, alpha=0.2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_file)
    plt.close()

def plot_metrics(data1, data2, figure_file, xlabel,ylabel):
    
    #assert len(data1) == len(data2)
    running_avg1 = np.zeros(len(data1))
    running_std1 = np.zeros(len(data1))

    running_avg2 = np.zeros(len(data2))
    running_std2 = np.zeros(len(data2))

    for i in range(len(data1)):
        running_avg1[i] = np.mean(data1[max(0, i-100):(i+1)])
        running_std1[i] = np.std(data1[max(0, i-100):(i+1)])

    for j in range(len(data2)):
        running_avg2[j] = np.mean(data2[max(0, j-100):(j+1)])
        running_std2[j] = np.std(data2[max(0, j-100):(j+1)])

    x1=[i+1 for i in range(len(data1))]
    plt.plot(x1, running_avg1,color='red',label="base_ppo")
    plt.fill_between(x1, running_avg1 - running_std1, running_avg1 + running_std1, color='red', alpha=0.2)

    x2=[i+1 for i in range(len(data2))]
    plt.plot(x2, running_avg2,color='green',label="ane_ppo")
    plt.fill_between(x2, running_avg2 - running_std2, running_avg2 + running_std2, color='green', alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_file)
    plt.close()

def multi_plotter(data, figure_file, xlabel, ylabel):
    color=['red','green','blue']
    label=['base_ppo','AnE_ppo(n=1)','AnE_ppo(n=5)']
    for j,data in enumerate(data):
        running_avg = np.zeros(len(data))
        running_std = np.zeros(len(data))

        for i in range(len(data)):
            running_avg[i] = np.mean(data[max(0, i-1000):(i+1)])
            running_std[i] = np.std(data[max(0, i-1000):(i+1)])


        x=[i+1 for i in range(len(data))]
        plt.plot(x, running_avg,label=label[j],color=color[j])
        #plt.fill_between(x, running_avg - running_std, running_avg + running_std, color=color[j], alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_file)
    plt.close()

def plot_bar_triplets(categories, values1, values2, values3, 
                      label1, label2, label3,
                      title, ylabel):
    
    x = np.arange(len(categories))  # label locations
    width = 0.25  # width of the bars

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, values1, width, label=label1,color='red')
    plt.bar(x,         values2, width, label=label2,color='green')
    plt.bar(x + width, values3, width, label=label3,color='blue')

    plt.xlabel('Environment')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('./plots/test_results.png')