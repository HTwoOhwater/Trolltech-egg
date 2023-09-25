import torch.nn as nn
import torch.optim as optim
import datetime
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from pylab import mpl
import cv2  # 这里要用滚动更新版，不然不能显示中文（旧版不支持UTF-8）
import os
import time


mpl.rcParams["font.sans-serif"] = ["SimHei"]


def showtime(start, end):
    time_cal = end - start
    result = ""
    second = int(time_cal % 60)
    time_cal = time_cal // 60
    minute = int(time_cal % 60)
    hour = int(time_cal // 60)
    if hour > 0:
        result = result + str(hour) + "h"
    if minute > 0:
        result = result + str(minute) + "min"
    if second > 0:
        result = result + str(second) + "s"
    return result


class Model:
    def __init__(self, model: nn.Module, params_path: str, ):
        self.model = model
        if params_path:
            self.model.load_state_dict(torch.load(f=params_path))
        self.data_set = None
        self.transform = None
        self.test_set = None

    def load_trans(self, __trans: transforms.transforms):
        self.transform = __trans

    def load_train_data(self, data_folder: str):
        self.data_set = datasets.ImageFolder(root=data_folder, transform=self.transform)

    def load_test_data(self, test_folder: str):
        self.test_set = datasets.ImageFolder(root=test_folder, transform=self.transform)

    def load_train_data_set(self, data_set):
        self.data_set = data_set

    def load_test_data_set(self, test_set):
        self.test_set = test_set

    def train(self, epochs: int, learning_rate: float, batch_sizes: int):
        dataloader = data.DataLoader(self.data_set, batch_size=batch_sizes, shuffle=True)
        optimizer = optim.Adam(params=self.model.parameters())
        criterion = nn.CrossEntropyLoss()

        time_start = time.time()

        for epoch in range(epochs):
            loss = None
            for index, (image, label) in enumerate(dataloader):
                result = self.model(image)
                loss = criterion(result, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            time_current = time.time()
            cost = showtime(time_start, time_current)
            print("Epoch:{}/{} , Time:{} , Loss: {}".format(epoch + 1, epochs, cost, loss))

    def save_params(self, path: str):
        torch.save(self.model.state_dict(), path)

    def predict(self, x):
        return self.model(x)

    def score(self):
        test_loader = data.DataLoader(dataset=self.test_set,
                                      batch_size=1,
                                      shuffle=False,
                                      )

        correct = 0
        total = 0

        with torch.no_grad():
            for image, label in test_loader:
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
                total += 1
                correct += (predicted == label).sum().item()

        accuracy = 100 * correct / total

        print("accuracy: {}".format(accuracy))




class MakeTrainer:
    def __init__(self, epochs: int, learning_rate: float, batch_size=64, model_name="", save_frequency=0, train_path=".core/train_libraries/", save_path=".core/model_versions/"):
        # 定义转换器
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 转化成224x224的图片，使其能输入模型
            transforms.ToTensor(),  # numpy数组张量化
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 对图片归一，加快训练效率
        ])
        self.save_frequency = save_frequency  # 保存频率
        self.epochs = epochs  # 训练次数
        self.learning_rate = learning_rate  # 学习率
        self.batch_size = batch_size  # 分批次训练，以节省内存，提升效率
        self.num_classes = 0  # 分类数（程序会自己处理）
        self.train_path = train_path  # 训练集位置（文件夹）
        self.path = save_path  # 保存模型位置（文件夹）
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # 加载预训练模型
        # 定义训练加载器
        self.train_dataset = datasets.ImageFolder(root=self.train_path, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        self.num_classes = len(self.train_dataset.classes)  # 根据你的文件夹个数，程序自动提供分类数目
        self.model.fc = nn.Linear(2048, self.num_classes)  # 将2048个输出维度转为分类个数个（即n分类）
        self.model.cpu()  # 有n卡的可以吧这个改成cuda()。就是这样做没有通用性，只能在n卡上面跑
        self.criterion = nn.CrossEntropyLoss()  # 定义损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 定义优化器
        self.model_name = model_name  # 君の名字

    def start(self):
        # 用来计算时间的函数

        start_time = time.time()  # 开始计时

        # 下面这一段是给模型分配它的文件夹
        if not self.model_name:  # 如果你没有填名字就是默认的日期时间的命名，自己去找，别记错了
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.path += current_time + "/"
        else:
            self.path += self.model_name + "/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # 下面这一大块都是记录你的文件夹名字和输出对应关系用的。文件夹的名字很关键！
        file = open(self.path + "keys_to_tensors.txt", mode="w+", encoding="UTF-8")
        file.write(str(self.train_dataset.class_to_idx))
        file.close()
        file = open(self.path + "keys.txt", mode="w+", encoding="UTF-8")
        keys = ",".join(self.train_dataset.classes)
        file.write(keys)
        file.close()
        # 终于要开始训练了
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images.unsqueeze(0)  # 增加维度，适应模型
                outputs = self.model(images)  # 计算当前输出
                loss = self.criterion(outputs, labels)  # 计算损失函数（和期望值的差距）
                self.optimizer.zero_grad()  # 清零梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 优化器操作
                # 计算时间差距用
                current_time = time.time()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Total time: '.format(epoch + 1, self.epochs, i + 1,
                                                                                       len(self.train_loader), loss.item()),
                      end="")
                print(showtime(start_time, current_time))
            # 能让你在训练过程中保存模型，前提是你填了保存频率
            if self.save_frequency != 0 and (epoch + 1) % self.save_frequency == 0:
                # 构造文件名
                filename = self.path + "/model" + str(epoch + 1) + ".pth"
                # 保存模型
                torch.save(self.model.state_dict(), filename)

        # 构造文件名
        filename = self.path + "/model.pth"
        # 保存模型
        torch.save(self.model.state_dict(), filename)
        # 返回字典值（说实话没啥用，我也不知道要干啥）
        return self.train_dataset.classes


class MakePredictor:
    def __init__(self, model_name: str, sensibility=2.0):
        # 摄像头占位
        self.cap = None
        # 模型路径
        self.model_name = ".core/model_versions/" + model_name
        # 模型输出维度
        self.num_classes = 0
        # 这里写了一个转换器，将输入的图片转化为224*224的RGB图片
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # 获取分类表
        file = open(self.model_name + "/keys.txt", mode="r", encoding="UTF-8")
        self.output_list = file.readline().split(",")
        self.num_classes = len(self.output_list)

        # 这个东西是敏感度。模型识别有一个阈值。敏感度高了准确但不易识别，低了易于识别但是不够精确。
        self.sensibility = sensibility

        # 是否输出原始维度信息
        if not self.output_list:
            self.raw_output_mode = True
        else:
            self.raw_output_mode = False

        # 从此番磁盘中加载模型到内存
        self.model = models.resnet50(num_classes=self.num_classes, weights=None)
        self.model.load_state_dict(torch.load(self.model_name + "/model.pth"))
        self.model.eval()

    # 读取单个图片预测
    def predict_image(self, image_path: str):
        # 这段是在读取图片
        image = plt.imread(image_path)
        image_show = image
        image = self.transform(image)
        output = []

        # 进行预测
        image = image.unsqueeze(0)
        result = self.model(image)
        value, index = torch.max(result, dim=1)

        if not self.output_list:
            print(result, value, index)
            output.append([result, value, index])
        else:
            plt.imshow(image_show)
            if value > self.sensibility:
                type_detected = self.output_list[index]
                text = "识别到了: {}".format(type_detected)
                # 在图表中添加文本注释
                plt.text(10, 10, text, bbox=dict(facecolor='red', alpha=0.8))
            else:
                text = "未能识别到目标"
                # 在图表中添加文本注释
                plt.text(10, 10, text, bbox=dict(facecolor='green', alpha=0.8))
        plt.show()
        return output

    # 读取一整个文件夹的图片预测
    def predict_folder(self, folder_path: str):
        # 这里是在获取文件目录
        test_dataset = datasets.ImageFolder(root=folder_path)
        output = []
        for image, label in test_dataset:
            image_show = image
            image = self.transform(image)

            # 进行预测
            image = image.unsqueeze(0)
            result = self.model(image)
            value, index = torch.max(result, dim=1)

            if not self.output_list:
                print(result, value, index)
                output.append([result, value, index])
            else:
                plt.imshow(image_show)
                if value > self.sensibility:
                    type_detected = self.output_list[index]
                    text = "识别到了: {}".format(type_detected)
                    # 在图表中添加文本注释
                    plt.text(10, 10, text, bbox=dict(facecolor='red', alpha=0.8))
                else:
                    text = "未能识别到目标"
                    # 在图表中添加文本注释
                    plt.text(10, 10, text, bbox=dict(facecolor='green', alpha=0.8))
            plt.show()
        return output

    # 在摄像头里预测
    def predict_camera(self, debug_mode=False):
        # 打开摄像头
        cap = cv2.VideoCapture(0)  # 参数0表示使用默认摄像头，如果有多个摄像头可以尝试不同的参数
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            print("无法打开摄像头")
            exit()
        # 循环读取和显示图像帧
        while True:
            # 从摄像头读取图像帧
            ret, frame = cap.read()
            # 检查图像帧是否成功读取
            if not ret:
                print("无法获取图像帧")
                break
            # 进行预测
            image = frame
            image = self.transform(image)
            image = image.unsqueeze(0)
            result = self.model(image)
            value, index = torch.max(result, dim=1)
            # debug模式能显示更多的细节，方便更好地评判模型性能
            if debug_mode:
                text = "最大张量值：" + str(value) + "\n最大张量：" + str(index) + "\n全部数值：" + str(result)
                if value > self.sensibility:
                    cv2.putText(frame, text, (10, 50), (0, 255, 0), cv2.FontFace("UTF8"), 20)
                else:
                    cv2.putText(frame, text, (10, 50), (0, 0, 255), cv2.FontFace("UTF8"), 20)
            else:
                if value > self.sensibility:
                    text = self.output_list[index[0]]
                    cv2.putText(frame, text, (10, 70), (0, 255, 0), cv2.FontFace("UTF8"), 50)
                else:
                    text = "未检测到对象"
                    cv2.putText(frame, text, (10, 70), (0, 0, 255), cv2.FontFace("UTF8"), 50)
            cv2.putText(frame, "按下Q键退出", (10, 20), (255, 0, 0), cv2.FontFace("UTF8"), 15)
            cv2.imshow("Frame", frame)
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) == ord('q'):
                break

        # 释放摄像头和关闭窗口
        cap.release()
        cv2.destroyAllWindows()


    def init_embedded(self, camera=0):
        # 打开摄像头
        self.cap = cv2.VideoCapture(camera)  # 参数0表示使用默认摄像头，如果有多个摄像头可以尝试不同的参数

    # 针对嵌入式的预测
    def predict_embedded(self, debug_mode=False):
        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            print("无法打开摄像头")
            exit()
        # 从摄像头读取图像帧
        ret, frame = self.cap.read()
        # 检查图像帧是否成功读取
        if not ret:
            print("无法获取图像帧")
            exit()
        # 进行预测
        image = frame
        image = self.transform(image)
        image = image.unsqueeze(0)
        result = self.model(image)
        value, index = torch.max(result, dim=1)
        if debug_mode:
            print(result, value, result)
        # 返回值是对应的标签。标签十分重要
        if value > self.sensibility:
            return self.output_list[index]
        else:
            return "None"

    # 这里突然想到一个绝妙的点子用来测试模型
    def test_model(self):
        pass
    # 可惜我没时间了


# 初始化的时候构建新目录
# 这个目录用来放测试集
if not os.path.exists(".core/data/test"):
    os.makedirs(".core/data/test")
# 这个目录用来放训练素材
if not os.path.exists(".core/data/train"):
    os.makedirs(".core/data/train")
# 这个目录用来放训练好的模型
if not os.path.exists(".core/model"):
    os.makedirs(".core/model")



