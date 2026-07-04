# KNN 有监督颜色识别说明

## 文件说明

KNN 版本新增了 3 个文件：

```text
color_recognition_knn.py
collect_knn_dataset.py
main_1_knn.py
```

训练集文件是：

```text
knn_color_dataset.csv
```

## 一、主程序如何调用 KNN？

`main_1_knn.py` 中使用下面这一行导入 KNN 颜色识别：

```python
from color_recognition_knn import color_classification
```

所以 KNN 文件也必须提供这个函数：

```python
def color_classification():
    ...
```

主程序不会给它传参数，调用方式仍然是：

```python
color_pred = color_classification()
```

## 二、KNN 颜色识别函数读取什么？

KNN 颜色识别函数读取当前目录下的 6 张图片：

```text
cube_test1.jpg
cube_test2.jpg
cube_test3.jpg
cube_test4.jpg
cube_test5.jpg
cube_test6.jpg
```

还会读取训练集：

```text
knn_color_dataset.csv
```

因此运行 KNN 主程序前，必须先采集训练集。

## 三、如何采集训练集？

采集训练集前，需要先把魔方复原，并放在机械结构的标准起始位置。

然后在目录 `/home/xx/Desktop/0617` 下运行：

```bash
QT_QPA_PLATFORM=offscreen python3 collect_knn_dataset.py
```

脚本会：

1. 拍摄 6 个面，保存为 `cube_test1.jpg` 到 `cube_test6.jpg`。
2. 按顺序给 6 个面打标签：`F U B D R L`。
3. 每个色块采集整块区域和 3x3 小区域特征。
4. 生成 `knn_color_dataset.csv`。

目前数据集规模是：

```text
540 条样本
```

## 四、KNN 使用了什么特征？

每个样本使用 7 个特征：

```text
r
g
b
h_cos
h_sin
s
v
```

其中：

```text
r/g/b 是归一化 RGB
h_cos/h_sin 是 HSV 色相 H 的圆周表示
s/v 是 HSV 的饱和度和亮度
```

使用 `h_cos` 和 `h_sin` 是为了避免色相角度 0 度和 360 度被误认为距离很远。

## 五、KNN 如何分类？

当前实现流程：

1. 读取训练集 `knn_color_dataset.csv`。
2. 读取 6 张待识别图片。
3. 对每个色块提取整块 + 3x3 小区域特征。
4. 对每个颜色类别计算 KNN 近邻平均距离。
5. 先识别物理颜色。
6. 根据本次拍到的 6 个中心块，重新映射成 `F/U/B/D/R/L`。
7. 强制最终结果中每个颜色正好 9 个，避免非法魔方状态。

## 六、颜色识别函数必须返回什么？

必须返回 54 个字符。

字符只能是：

```text
F U B D R L
```

顺序必须是：

```text
F U B D R L
```

每面内部顺序是：

```text
左上  上中  右上
左中  中心  右中
左下  下中  右下
```

## 七、如何单独测试 KNN？

先保证当前目录下有：

```text
cube_test1.jpg 到 cube_test6.jpg
knn_color_dataset.csv
```

然后运行：

```bash
python3 - <<'PY'
from collections import Counter
from color_recognition_knn import color_classification

result = list(color_classification())
print(''.join(result), len(result))
print(Counter(result))
PY
```

正确结果应该满足：

```text
长度是 54
F/U/B/D/R/L 每类都是 9 个
```

## 八、如何运行完整 KNN 魔方复原？

运行：

```bash
QT_QPA_PLATFORM=offscreen python3 main_1_knn.py
```

如果当前魔方已经是复原状态，求解步骤可能为空，这是正常现象。

## 九、学员可以改哪里？

学员主要修改：

```text
color_recognition_knn.py
```

可以改：

```text
特征提取方式
K 值
距离计算方式
投票方式
后处理方式
```

但是必须保持：

```python
def color_classification():
    ...
```

并且最终返回 54 个合法颜色字符。
