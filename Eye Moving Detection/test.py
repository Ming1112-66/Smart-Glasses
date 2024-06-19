import torch
from torch.utils.data import DataLoader
from train import EmdDataset
import itertools
import matplotlib.pyplot as plt
from torchvision.utils import Image
from PIL import Image

plt.rc("font", family="DengXian")
import numpy as np
from serial import Serial
from serial.tools import list_ports
from to_matrix import MatrixPipeline
from time import sleep
import json
import os

CAR_OP = {
    "前进": 1,
    "后退": 2,
    "左移": 3,
    "右移": 4,
    "左转": 5,
    "右转": 6,
    "停止": 7,
}




def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比")
        np.set_printoptions(formatter={"float": "{: 0.2f}".format})
        # print(cm)
    else:
        print("显示具体数字")
        # print(cm)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

def visualize():
    model: torch.nn.Module = torch.load("model.pth", map_location=torch.device('cpu'))
    model.eval()
    dataset = EmdDataset(cuda=False)
    dataloader = DataLoader(dataset, 1, False)
    num_labels = len(dataset.labels)
    cm = torch.zeros(num_labels, num_labels)
    for img, label in dataloader:
        pred = model(img)
        pred_index = torch.argmax(pred, dim=1)
        gt_index = torch.argmax(label, dim=1)
        cm[gt_index, pred_index] += 1
    print("acc:", torch.sum(torch.einsum('ii -> i', cm)) / dataset.len)
    plot_confusion_matrix(cm.numpy(), dataset.labels, True)

def read_mapping_config(path="mapping.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path} 文件")
    try:
        mapping = json.load(open(path, 'r', encoding='utf-8'))
        return mapping
    except Exception as e:
        raise e

def prompt_mapping(labels: list[str]):
    try:
        mapping = read_mapping_config()
        p = input("已读取存在的配置文件。是否使用？[Y/n]:")
        print(mapping)
        if p in ['Y', 'y', '']:
            return {int(k): v for k, v in mapping.items()}
    except FileNotFoundError:
        print("找不到配置文件，将重新要求配置映射。")
    except Exception as e:
        print(f"读取配置文件出错：{e}")
    print("请按照下面的格式输入映射。")
    lines = [f"{label} -> {i}" for i, label in enumerate(labels)]
    print(*lines, sep='\n')
    print("请输入小车指令对应的数字。如果没有对应，直接回车输入空即可。例如，刚刚输出了 上看 -> 0 ，那么在需要上看对应的小车操作中输入0。")
    mapping = dict.fromkeys(CAR_OP.values(), '')
    for op, i in CAR_OP.items():
        mapping[i] = input(f"{op}:")
    # reverse mapping
    mapping = {int(v): k for k, v in mapping.items() if len(v)}
    json.dump(mapping, open("mapping.json", 'w', encoding='utf-8'), ensure_ascii=False)
    return mapping


class MovingWindow:
    def __init__(self, size=8) -> None:
        self.window = []
        self.size = size
    
    def push(self, item):
        self.window.append(item)
        if len(self.window) > self.size:
            self.window.pop(0)



TEST_DATA = "A,254,205,223,157,253,205,223,157,212,224,168,143,211,227,167,259,184,156,266,242,248,258,281,257,180,153,265,242,249,261"
TEST = False
OUTPUT = True



if __name__ == "__main__":
    visualize()
    mw = MovingWindow(size=16)
    if not TEST:
        available_ports = list_ports.comports()
        out_port = input(
            f"请输入数据串口号(可用串口：{', '.join(list(map(lambda x: x.device, available_ports)))}):"
        )
        out_serial = Serial(out_port, 115200, timeout=0.5)
        if OUTPUT:
            in_port = input("请输入输出串口号：")
            in_serial = Serial(in_port, 115200, timeout=.5)
        print("loaded serial ports")
    model: torch.nn.Module = torch.load("model.pth", map_location=torch.device('cpu'))
    model.eval()
    dataset = EmdDataset(cuda=False)
    mapping = prompt_mapping(dataset.labels)
    print("loaded nn modules")
    while True:
        line = out_serial.readline().decode("utf-8") if not TEST else TEST_DATA
        # print(line)
        try:
            pipeline = MatrixPipeline()
            pipeline.data = [line]
            data = pipeline.to_array((0, 1)).transform_data(400).data[0]
            data = Image.fromarray(data * 255., mode="L")
            data = torch.from_numpy(np.array(data)).unsqueeze(0).unsqueeze(0) / 255.
            # print(data)
            pred = model(data)
            pred_label = torch.argmax(pred, dim=1)
            mw.push(pred_label.numpy()[0])
            # pick the most frequent label
            label = np.argmax(np.bincount(mw.window))
            car_op = str(mapping[label])
            print(dataset.labels[label], car_op)
            if OUTPUT:
                in_serial.write(bytes(car_op, 'utf-8'))
            sleep(0.03)
            if TEST:
                break
        except:
            continue
