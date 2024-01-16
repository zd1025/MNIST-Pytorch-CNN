![1705385899163](https://github.com/zd1025/MNIST-Pytorch-CNN/assets/87258436/7fb8be05-f04d-4543-9637-d1e0a69d7427)![1705385899163](https://github.com/zd1025/MNIST-Pytorch-CNN/assets/87258436/3756436a-fb9b-4ce2-92bd-369286131615)# MNIST-Pytorch-CNN
手写识别MNIST（Pytorch + CNN实现）

[TOC]



## 一、项目启动

```
项目文件树
```

<img src="https://common-1316603156.cos.ap-shanghai.myqcloud.com/public/image-20230502101523577.png" style="zoom:80%;" />

```
可能出现的报错
```

- **Could not build wheels for opencv-python which use PEP 517 and cannot be installed directly**

  - ```
    关于cv2安装报错问题解决方案（anaconda下）->->->
    前景：本人的python环境是3.7 因此会出现如下错误
    报错：Could not build wheels for opencv-python which use PEP 517 and cannot be installed directly
    原因：cv2存在于opencv-python库中 直接进行pip出现报错
    解决方案：  1、前往网站 https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv 下载对应 python 版本的 whl 文件
               2、放到本地 python 文件下的Lib文件夹中
               3、打开 cmd 进入whl所在文件夹下的路径 输入命令行 pip install whl文件名
               
    ```

- **ImportError: numpy.core.multiarray failed to import**

  - ```
    关于 opencv-python 与 numpy 不兼容的解决方案->->->
    报错：ImportError: numpy.core.multiarray failed to import
    解决方案：  1、pip uninstall numpy 卸载旧版本 numpy
               2、pip install numpy 最新版本 numpy 已经能适配
    ```

**博文参考链接：**```https://www.jb51.net/article/171584.htm```



## 二、网络结构

```
网络整体结构图
```
![1705385915528](https://github.com/zd1025/MNIST-Pytorch-CNN/assets/87258436/f414ff0a-9df4-4152-a7e2-1997e50ffbdd)



## 三、结果展示

```
训练时的loss曲线
```

<img src="https://secure2.wostatic.cn/static/eNGxRuFacUAFxXee9CVP9U/image.png?auth_key=1682994658-4pyCR7xEHotRTJy9CgW1Hq-0-ffe1f84ca08ddfa5071efecf738a0cd3&file_size=19419" alt="https://secure2.wostatic.cn/static/eNGxRuFacUAFxXee9CVP9U/image.png?auth_key=1682994658-4pyCR7xEHotRTJy9CgW1Hq-0-ffe1f84ca08ddfa5071efecf738a0cd3&file_size=19419" style="zoom:50%;" />

```
测试曲线（测试Loss曲线 / 测试的正确率曲线）
```

<img src="https://secure2.wostatic.cn/static/6K86VNbxHnowTNkMEA6tVo/image.png?auth_key=1682994704-sgYQ3H4iUDfQATrKmvfsE9-0-620d525f8530a8bff8eb9a31c946501a&file_size=16119" alt="https://secure2.wostatic.cn/static/6K86VNbxHnowTNkMEA6tVo/image.png?auth_key=1682994704-sgYQ3H4iUDfQATrKmvfsE9-0-620d525f8530a8bff8eb9a31c946501a&file_size=16119" style="zoom:50%;" />

<img src="https://secure2.wostatic.cn/static/a1hgshLmu4YTBpyWnHQamT/image.png?auth_key=1682994725-ak6dGwvNdxd7BnC5j9ihKG-0-925f4cdcd0ad924e2c2ea26236cfcc5f&file_size=19651" alt="https://secure2.wostatic.cn/static/a1hgshLmu4YTBpyWnHQamT/image.png?auth_key=1682994725-ak6dGwvNdxd7BnC5j9ihKG-0-925f4cdcd0ad924e2c2ea26236cfcc5f&file_size=19651" style="zoom:50%;" />



## 四、附上笔记链接

MNIST手写识别

https://www.wolai.com/rvoV22eCuW9u4TnZVMGqtk
