import numpy as np
import pandas as pd
from PIL import Image
import os
import shutil

def transform_data(data: np.ndarray):
    mat = np.zeros((5, 6), dtype=np.float32)
    data_l: np.ndarray = data[:15].reshape((5, 3))
    mat[:, :3] = data_l
    data_r: np.ndarray = data[15:].reshape((5, 3))
    mat[:, 3:] = np.flip(data_r, axis=1)
    return mat

def prompt_pipeline_parameters():
    note = '''
数据文件夹应具有这样的格式：
data/
    1/ (1代表第一组实验)
        上看1.xlsx （每一个xlsx文件以“标签名”+“实验组号”.xlsx命名）
        下看1.xlsx
        ...
    ...
而输出的图片文件夹根据上述格式，具有如下格式：
images/
    上看/ （每一个标签名）
        1_0.jpg （每一张图片以“实验组号_序号”.jpg命名）
        1_1.jpg
        ...
'''
    print(note)
    excel_path = input("请输入数据文件夹相对该程序的位置（按下Enter,则默认为该目录下的data/）：")
    if excel_path == "":
        excel_path = "data/"
    range = input("请输入需要读取数据的行数范围，开始的数字和结束的数字以空格分割（按下Enter,则默认去除头200行和尾200行的数据，只保留中间数据）：")
    if range == "":
        range = (200, -200)
    else:
        range = tuple([int(i) for i in range.split()])
    window_size = input("请输入需要进行移动平均处理的窗口大小（按下Enter,则默认为8）：")
    if window_size == "":
        window_size = 8
    else:
        window_size = int(window_size)
    u = input("所有样本需要除以一个统一的值u，统一处理到[0, 1]范围内，方便算法处理。请输入这个值u（按下Enter,则默认为400）：")
    if u == "":
        u = 400
    else:
        u = float(u)
    return {
        "excel_path": excel_path,
        "range": range,
        "window_size": window_size,
        "u": u
    }


class MatrixPipeline:
    def __init__(self) -> None:
        pass

    def read_excel(self, path: str):
        print(f"Reading {path}")
        data = pd.read_excel(path)
        for d in data.columns:
            self.data = list(data[d])  # returns the first column
            return self

    def to_array(self, range: tuple[int, int]):
        data = self.data[range[0]:range[1]]
        res = []
        for i, d in enumerate(data):
            if not isinstance(d, str):
                continue
            try:
                arr = d.split(',')[1:]  # remove "A,"
                arr = [float(a) for a in arr]
                assert len(arr) == 30
                res.append(np.array(arr, dtype=np.float32))
            except:
                continue
        self.data = np.array(res)
        return self

    def sample_transform(self):
        sample_data = np.linspace(1, 30, 30)
        print(transform_data(sample_data))
        return self

    def conv1d(self, window_size: int):
        data = np.copy(self.data)
        self.data = np.zeros((data.shape[0] - window_size + 1, 30))
        for i, d in enumerate(data[window_size - 1:]):
            window = data[i:i + window_size]
            self.data[i] = np.mean(window, axis=0)
        return self

    def transform_data(self, u: float):
        data = np.copy(self.data / u)  # len(data), 30
        self.data = np.zeros((data.shape[0], 5, 6), dtype=np.float32)
        for i, d in enumerate(data):
            # apply the transformation rule
            self.data[i] = transform_data(d)
        return self

    def write_data(self, path: str, fname: str):
        for i, d in enumerate(self.data):
            img = Image.fromarray(d * 255., mode="L")
            img.save(f"{path}/{fname}_{i}.jpg")


if __name__ == "__main__":
    print("注意！请确保当前程序所在文件夹下，没有images/文件夹，或者它是空的！")
    if os.path.exists("images/"):
        shutil.rmtree("images/", ignore_errors=True)
        print("rm -rf images/")
    params = prompt_pipeline_parameters()
    print(params)
    pipeline = MatrixPipeline()
    if not os.path.exists("images/"):
        os.mkdir("images/")
    for i in os.listdir(params["excel_path"]):
        for excel in os.listdir(os.path.join(params["excel_path"], i)):
            label = excel[:-6] # label1.xlsx -> label1
            path = f"images/{label}"
            if not os.path.exists(path):
                os.mkdir(path)
            pipeline.read_excel(os.path.join(params["excel_path"], i, excel)) \
                .to_array(params["range"]) \
                .conv1d(params["window_size"]) \
                .transform_data(params["u"]) \
                .write_data(path, i)
        
