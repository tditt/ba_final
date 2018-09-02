import matplotlib.pyplot as plt
import numpy as np
from os import listdir

path = 'trained_retinanet_models/logs/'


def read_log(path, file):
    text_file = open(path + file, 'r')
    all_lines = text_file.readlines()
    text_file.close()
    epochs_step = True
    all_epochs = []
    for i, line in enumerate(all_lines):
        if epochs_step:
            epoch = {}
            epochs_step = False
        n = i % 16
        if n == 7:
            epoch['false_positives'] = getBracketsValue(line)
        elif n == 8:
            epoch['precision'] = getColonValue(line)
        elif n == 9:
            epoch['recall'] = getColonValue(line)
        elif n == 10:
            epoch['f1_score'] = getColonValue(line)
        elif n == 12:
            ap_values = getAPs(line)
            epoch['map'] = ap_values[0]
            epoch['ap50'] = ap_values[1]
            epoch['ap75'] = ap_values[2]
            all_epochs.append(epoch)
            epochs_step = True
    return all_epochs


def getBracketsValue(line):
    return float(line.split("(")[1].split(" ")[0])


def getColonValue(line):
    return float(line.split(": ")[1])


def getAPs(line):
    arr = line.split(": ")
    map = float(arr[1].split(" ")[0])
    ap50 = float(arr[2].split(" ")[0])
    ap75 = float(arr[3])
    return [map, ap50, ap75]


def draw_graph(all_epochs, metric, label, title, color):
    n = len(all_epochs)
    x = np.arange(n)
    y = np.zeros(n)
    for i, epoch in enumerate(all_epochs):
        y[i] = epoch[metric]
    plt.plot(x, y, color)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(path + metric + ".png")
    plt.show()


def draw_graph(all_epochs, metric, label, title, color, filename):
    n = len(all_epochs)
    x = np.arange(n)
    y = np.zeros(n)
    for i, epoch in enumerate(all_epochs):
        y[i] = epoch[metric]
    plt.plot(x, y, color)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(path + filename + ".png")
    plt.show()


def draw_multi_graph(all_epochs, metrics, labels, title, colors, filename):
    n = len(all_epochs)
    x = np.arange(n)
    y = np.zeros(n)
    for j, metric in enumerate(metrics):
        for i, epoch in enumerate(all_epochs):
            y[i] = epoch[metric]
        plt.plot(x, y, color=colors[j], label=labels[j])

    plt.xlabel('epoch')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(path + filename + ".png")
    plt.show()


def create_plots_seperate(file):
    all_epochs = read_log(path, file)
    draw_graph(all_epochs, 'false_positives', 'false positives', 'False Positives curve over all epochs (IOU=0.5)',
               'red', file + '_false_positives')
    draw_graph(all_epochs, 'precision', 'precision', 'Precision curve over all epochs (IOU=0.5)', 'blue',
               file + '_precision')
    draw_graph(all_epochs, 'recall', 'recall', 'Recall curve over all epochs (IOU=0.5)', 'cyan', file + '_recall')
    draw_graph(all_epochs, 'f1_score', 'f1 score', 'F1-Score curve over all epochs (IOU=0.5)', 'purple',
               file + '_f1_score')
    draw_graph(all_epochs, 'map', 'mAP', 'mAP curve over all epochs', 'xkcd:lime green', file + '_map')
    draw_graph(all_epochs, 'ap50', 'AP50', 'AP curve over all epochs (IOU=0.5)', 'green', file + '_ap50')
    draw_graph(all_epochs, 'ap75', 'AP75', 'AP curve over all epochs (IOU=0.75)', 'olive', file + '_ap75')


def create_plots_multi(file):
    all_epochs = read_log(path, file)
    labels = ['False Positive Rate', 'Precision', 'Recall', 'F1-Score']
    metrics = ['false_positives', 'precision', 'recall', 'f1_score']
    colors = ['red', 'blue', 'cyan', 'purple']
    draw_multi_graph(all_epochs, metrics, labels, 'Metrics curves over all epochs (IOU=0.5)', colors, file + '_metrics')
    labels = ['mAP', 'AP50', 'AP75']
    metrics = ['map', 'ap50', 'ap75']
    colors = ['xkcd:lime green', 'green', 'olive']
    draw_multi_graph(all_epochs, metrics, labels, 'AP-Metrics curves over all epochs', colors, file + '_ap_metrics')


for file in listdir(path):
    create_plots_multi(file)