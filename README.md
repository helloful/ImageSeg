# ImageSeg

- 本Repo是《医学图像分析》课程的作业代码

# 代码使用流程

1、使用utils.py中 create_dataset函数创建训练文件，具体根据自己的数据集路径修改代码设置

2、训练Unet模型

```
python train_model1.py
```

3、验证Unet模型

```
python test_model1.py
```

4、训练Inf-Net的修改模型

```
python train_model2.py
```

5、验证Inf-Net模型

```
python test_model2.py
```

6、半监督的训练和验证，函数均在semi_train.py文件中，根据训练和测试调用不同的函数
