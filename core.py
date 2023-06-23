import torch.nn as nn
import torch.optim as optim
import datetime
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pylab import mpl
import cv2  # 这里要用滚动更新版，不然不能显示中文（旧版不支持UTF-8）
import os
import urllib
import re
from bs4 import BeautifulSoup
import time

mpl.rcParams["font.sans-serif"] = ["SimHei"]


class MakeTrainer:
    def __init__(self, epochs: int, learning_rate: float, batch_size=64, model_name="", save_frequency=0, train_path=".core/train_libraries/", save_path=".core/model_versions/"):
        self.save_frequency = save_frequency  # 保存频率
        self.epochs = epochs  # 训练次数（一般5到10就行）
        self.learning_rate = learning_rate  # 学习率（一般1e-4就能应付大部分情况了）
        self.batch_size = batch_size  # 分批次训练，以节省内存，提升效率，默认就好，一般不需要改
        self.num_classes = 0  # 分类数（程序会自己处理）
        self.train_path = train_path  # 训练集位置（文件夹）
        self.path = save_path  # 保存模型位置（文件夹）
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # 加载预训练模型
        # 定义转换器
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 转化成224x224的图片，使其能输入模型
            transforms.ToTensor(),  # numpy数组张量化
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 对图片归一，加快训练效率
        ])
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


class MakeCrawler:
    def __init__(self):
        self.header = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 '
                'UBrowser/6.1.2107.204 Safari/537.36'
        }
        self.url = "https://cn.bing.com/images/async?q={0}&first={1}&count={" \
              "2}&scenario=ImageBasicHover&datsrc=N_I&layout=ColumnBased&mmasync=1&dgState=c*9_y" \
              "*2226s2180s2072s2043s2292s2295s2079s2203s2094_i*71_w*198&IG=0D6AD6CBAF43430EA716510A4754C951&SFX={" \
              "3}&iid=images.5599"

    def start(self, key_word: str, crawl_number=100, save_path=".core/image_assets/", crawled_number=0, fast_train=False):
        def get_image(url, count, path):
            try:
                time.sleep(0.5)
                urllib.request.urlretrieve(url, path + str(count + 1) + '.jpg')
            except Exception as e:
                time.sleep(1)
                print("图像因为未知原因无法保存，正在跳过...")
                return count
            else:
                print("图片+1，已保存 " + str(count + 1) + " 张图")
                return count + 1

        # ----------------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------------
        # 找到原图并返回URL
        def findImgUrlFromHtml(html, rule, url, key, first, loadNum, sfx, count, path, crawl_number):
            soup = BeautifulSoup(html, "lxml")
            link_list = soup.find_all("a", class_="iusc")
            url = []
            for link in link_list:
                result = re.search(rule, str(link))
                # 将字符串"amp;"删除
                url = result.group(0)
                # 组装完整url
                url = url[8:len(url)]
                # 打开高清图片网址
                count = get_image(url, count, path)
                if count >= crawl_number:
                    break
            # 完成一页，继续加载下一页
            return count

        # ----------------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------------
        # 获取缩略图列表页
        def getStartHtml(url, key, first, loadNum, sfx):
            # 打开页面
            page = urllib.request.Request(url.format(key, first, loadNum, sfx), headers=self.header)
            html = urllib.request.urlopen(page)
            return html

        if fast_train:
            save_path = ".core/train_libraries/"
        save_path += key_word + "/"
        # 将关键词转化成URL编码
        key = urllib.parse.quote(key_word)
        # URL中的页码（这里的页码有些抽象，指的是从第几个图片开始）
        first = 1
        # URL中每页的图片数
        loadNum = 35
        # URL中迭代的图片位置（即每页第几个图片）
        sfx = 1
        # 用正则表达式去匹配图片的URL
        rule = re.compile(r"\"murl\":\"http\S[^\"]+")
        # 没有目录就创建目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 开始爬取
        print("初始化成功！正在爬取。")
        while crawled_number < crawl_number:
            # 获取当前页的html内容
            html = getStartHtml(self.url, key, first, loadNum, sfx)
            # 获取图片的URL并保存图片
            crawled_number = findImgUrlFromHtml(html, rule, self.url, key, first, loadNum, sfx, crawled_number, save_path,
                                                crawl_number)
            # 防止爬取之前的图片
            first = crawled_number + 1

            sfx += 1
        print("爬取成功！已经完成关键词为{0:}的图片爬取{1:}张".format(key_word, crawl_number))


# 初始化的时候构建新目录
if not os.path.exists(".core/image_assets"):
    os.makedirs(".core/image_assets")
# 这个目录用来放爬虫的爬取的内容
if not os.path.exists(".core/train_libraries"):
    os.makedirs(".core/train_libraries")
# 这个目录用来放训练素材
if not os.path.exists(".core/model_versions"):
    os.makedirs(".core/model_versions")
# 这个目录用来放训练好的模型
if not os.path.exists(".core/test_sets"):
    os.makedirs(".core/test_sets")
# 这个目录用来放测试用的图片（如果你喜欢放在里面的话）
