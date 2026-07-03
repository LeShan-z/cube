# 颜色识别算法编写注意事项

## 一、文件作用

学员只需要编写或修改这个文件：

```text
color_recognition_algorithm.py
```

主程序文件：

```text
main_1.py
```

会通过下面这一行导入颜色识别算法：

```python
from color_recognition_algorithm import color_classification
```

所以学员必须在 `color_recognition_algorithm.py` 中提供一个函数：

```python
def color_classification():
    ...
```

## 二、主程序会传什么参数？

主程序不会给 `color_classification()` 传入任何参数。

也就是说，函数必须保持下面这种形式：

```python
def color_classification():
    ...
```

不要写成：

```python
def color_classification(images):
    ...
```

也不要写成：

```python
def color_classification(cam):
    ...
```

因为 `main_1.py` 中调用方式是：

```python
color_pred = color_classification()
```

## 三、颜色识别函数从哪里拿数据？

主程序拍照后，会在当前运行目录下生成 6 张图片：

```text
cube_test1.jpg
cube_test2.jpg
cube_test3.jpg
cube_test4.jpg
cube_test5.jpg
cube_test6.jpg
```

学员的颜色识别算法需要自己读取这 6 张图片。

示例：

```python
import cv2

for i in range(1, 7):
    image = cv2.imread("cube_test%d.jpg" % i)
```

## 四、颜色识别函数需要返回什么？

`color_classification()` 必须返回 54 个颜色字符。

可以返回：

```python
list
```

也可以返回：

```python
numpy.ndarray
```

但里面必须是 54 个字符，每个字符只能是下面 6 个之一：

```text
F U B D R L
```

含义是：

```text
F: Front，前面
U: Up，上面
B: Back，后面
D: Down，下面
R: Right，右面
L: Left，左面
```

返回示例：

```python
return [
    "F", "F", "F", "F", "F", "F", "F", "F", "F",
    "U", "U", "U", "U", "U", "U", "U", "U", "U",
    "B", "B", "B", "B", "B", "B", "B", "B", "B",
    "D", "D", "D", "D", "D", "D", "D", "D", "D",
    "R", "R", "R", "R", "R", "R", "R", "R", "R",
    "L", "L", "L", "L", "L", "L", "L", "L", "L",
]
```

## 五、54 个颜色的顺序

返回的 54 个颜色必须按下面顺序排列：

```text
F U B D R L
```

也就是：

```text
第 1-9 个：F 面
第 10-18 个：U 面
第 19-27 个：B 面
第 28-36 个：D 面
第 37-45 个：R 面
第 46-54 个：L 面
```

每一面的 9 个色块顺序是：

```text
左上  上中  右上
左中  中心  右中
左下  下中  右下
```

对应索引：

```text
0 1 2
3 4 5
6 7 8
```

## 六、必须保证每种颜色正好 9 个

最终返回结果中：

```text
F 必须有 9 个
U 必须有 9 个
B 必须有 9 个
D 必须有 9 个
R 必须有 9 个
L 必须有 9 个
```

如果某个颜色不是 9 个，后面的魔方求解可能失败。

## 七、中心块很重要

魔方每一面的中心块决定这一面的颜色名称。

每面中心块的位置是第 5 个，也就是局部索引 `4`。

全局中心块索引分别是：

```text
F 面中心：4
U 面中心：13
B 面中心：22
D 面中心：31
R 面中心：40
L 面中心：49
```

颜色识别时建议优先利用中心块来确定每一面的颜色类别。

## 八、推荐算法思路

可以按下面流程写：

1. 读取 `cube_test1.jpg` 到 `cube_test6.jpg`。
2. 在每张图中固定裁剪 9 个小色块区域。
3. 对每个小色块计算 RGB 或 HSV 的平均值。
4. 根据中心块颜色确定 6 个颜色类别。
5. 对其他色块分类到最近的中心颜色。
6. 输出长度为 54 的颜色序列。

当前参考实现使用的是：

```text
RGB 均值 + HSV 转换 + 白色低饱和度识别 + 色相聚类
```

学员也可以使用其他方法，例如：

```text
RGB 欧氏距离
HSV 阈值
K-means 聚类
手写规则分类
```

只要最终返回格式正确即可。

## 九、不要在颜色识别文件里做这些事

不要控制机械臂。

不要写串口指令。

不要打开摄像头。

不要调用魔方求解算法。

不要修改 `main_1.py`。

颜色识别文件只负责一件事：

```text
读取 6 张图片，识别 54 个色块，返回颜色序列。
```

## 十、最小模板

学员可以从这个模板开始：

```python
import cv2
import numpy as np


def color_classification():
    result = []

    for i in range(1, 7):
        image = cv2.imread("cube_test%d.jpg" % i)
        if image is None:
            raise FileNotFoundError("cube_test%d.jpg not found" % i)

        # TODO: 在这里识别当前面的 9 个色块
        # 必须按照左上、上中、右上、左中、中心、右中、左下、下中、右下的顺序加入 result

    if len(result) != 54:
        raise ValueError("color_classification() must return 54 colors")

    for color in result:
        if color not in ["F", "U", "B", "D", "R", "L"]:
            raise ValueError("invalid color: " + str(color))

    return result
```

## 十一、单独测试颜色识别文件

在 `/home/xx/Desktop/0617` 目录下，可以用下面命令单独测试：

```bash
python3 -c "from color_recognition_algorithm import color_classification; print(''.join(color_classification()))"
```

如果输出一个 54 位字符串，并且没有报错，说明颜色识别文件可以被主程序调用。

再运行完整主程序：

```bash
QT_QPA_PLATFORM=offscreen python3 main_1.py
```
