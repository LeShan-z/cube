# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

魔方机器人解魔方系统：摄像头拍摄魔方6个面 → 颜色识别 → Two-Phase算法求解 → 串口指令控制机械臂复原。

## 入口与运行

两个主入口功能等价，区别在颜色识别模块：

```bash
# HSV聚类版
python main_1.py

# KNN版
python main_1_knn.py

# 无GUI环境（如SSH远程）
QT_QPA_PLATFORM=offscreen python main_1.py

# 单独测试颜色识别
python -c "from color_recognition_algorithm import color_classification; print(''.join(color_classification()))"
python -c "from color_recognition_knn import color_classification; print(''.join(color_classification()))"
```

## 硬件依赖

- 摄像头：OpenCV `VideoCapture(0)`，实时预览画面中央 480×480 区域
- 机械臂：串口 `/dev/ttyUSB0` @115200 baud，协议为 `#` 开头 `!` 结尾的ASCII指令
- 灯光控制：`#X!` 开灯、`#Y!` 关灯、`#=!` / `#-!` 调亮度
- 该系统运行在 Linux 上（串口路径、`QT_QPA_PLATFORM=offscreen`）

## 核心流程

### 1. 拍照采集 (`main_1.py` 的 `recognize_cube`)
摄像头预览 → 按 `s` 键或自动触发 → 依次拍摄6个面，每拍完一面通过串口翻动魔方（`#BU!` / `#BUM!` / `#MBU!` / `#IBU!` / `#BBUM!`），保存为 `cube_test1.jpg` ~ `cube_test6.jpg`（顺序：F U B D R L）。

### 2. 颜色识别

两种实现，接口相同：`color_classification() -> list/ndarray[str]`，返回54个字符（6面×9色块），每面按左上→右下顺序。

**`color_recognition_algorithm.py`（HSV聚类）**：
- 按固定坐标(480×480图中9个60×60贴片区域)裁剪色块
- RGB均值 → HSV转换 → 白色块通过最低S值分离 → 其余5色用色相(H)做圆形K-means聚类

**`color_recognition_knn.py`（KNN）**：
- 从 `knn_color_dataset.csv` 加载训练数据（7维特征：R, G, B, cos(H), sin(H), S, V）
- 对每面9个色块各采样10个子区域（整体+3×3网格），取KNN最近邻投票
- 通过中心块排列搜索确定物理颜色到面标签的映射，再平衡到每色恰好9个

`collect_knn_dataset.py`：从已还原魔方拍摄的6张图中采集训练数据，生成/更新 `knn_color_dataset.csv`。

### 3. 求解 (`twophase/`)

标准Kociemba Two-Phase Algorithm实现：

| 模块 | 职责 |
|---|---|
| `twophase/__init__.py` → `solve()` | 对外唯一接口，接收54字符颜色串（U R F D L B顺序），返回解法字符串如 `"U L2 F R' D"` |
| `pieces.py` | `Color`, `Corner`, `Edge`, `Facelet` 枚举定义 |
| `cubes/facecube.py` | 面层表示（54 sticker → 颜色），提供 `to_cubiecube()` 转换 |
| `cubes/cubiecube.py` | 块层表示（角块/棱块的排列+朝向），所有坐标属性计算和6种基本转动 |
| `cubes/coordcube.py` | 坐标紧凑表示，直接从坐标值构造无需遍历 |
| `solve.py` | `SolutionManager` — IDA* 双向搜索 |
| `tables.py` | 移动表和剪枝表，首次加载从 `tables.json` 读取，若无文件则计算并写入 |
| `random.py` | 生成随机合法魔方状态 |

**Phase 1**：搜索到 G1 = ⟨U, D, R2, L2, F2, B2⟩ 子群，坐标用 twist(3⁷)、flip(2¹¹)、udslice(₁₂C₄)。  
**Phase 2**：在 G1 内搜索到还原态，坐标用 corner(8!)、edge4(4!)、edge8(8!)。  
剪枝表：udslice×twist + udslice×flip（Phase1），edge4×edge8 + edge4×corner（Phase2）。

### 4. 指令转换与执行

`solve()` 返回标准魔方符号（如 `U R2 F'`） → `step_transformation()` 转换为机械臂步骤码 → `step_run()` 编码为串口指令字符串 `#BUJ...U!` → 串口发送执行。

## 颜色识别接口约定

来自 `COLOR_RECOGNITION_GUIDE.md` 的硬性约束：
- 函数签名必须是 `color_classification()`，无参数
- 自行读取当前目录下 `cube_test1.jpg` ~ `cube_test6.jpg`
- 返回54个字符，顺序 F→U→B→D→R→L，每面左上→右上→左下→右下
- 每色必须恰好9个（F/U/B/D/R/L各9），否则求解失败
- 中心块位置（全局索引 4, 13, 22, 31, 40, 49）决定该面颜色标签
- 该文件不应操作摄像头、串口、机械臂，只负责纯图像识别

## 注意事项

- `tables.json` 约21MB，首次运行若无此文件会自动生成（耗时较长）
- `main_1.py` 中 `auto_start_once = True` 会使首次循环自动模拟按 `s` 键，跳过手动触发
- `data` 文件是二进制数据（可能是旧版预计算表或测试数据）
- `main_1_before_split.py` 和 `.bak_before_autostart` 是备份文件，不是当前使用的版本
- `twophase.__init__.py` 还提供了 `solve_best()` 和 `solve_best_generator()` 用于迭代寻找更优解
