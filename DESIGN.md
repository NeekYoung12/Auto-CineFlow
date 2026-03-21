## 📄 第一部分：需求文档 (PRD)

### 1. 项目目标
构建一个由 LLM 驱动的影视分镜参数生成器。输入一段 1-2 人的故事叙述，输出符合电影工业标准的结构化指令集（JSON），用于驱动后续的 AI 绘图与视频工作流。

### 2. 核心用户场景 (User Stories)
* **场景 A：** 用户输入“两人在酒馆对坐”，系统需自动生成 Master Shot（全景）建立空间，并锁定两人的左右位置。
* **场景 B：** 角色情绪爆发时，系统需自动识别并切换至 Close-up（特写），同时调整焦距参数和构图重心。

### 3. 功能需求清单
* **[REQ-01] 角色状态机：** 自动维护 1-2 个角色的 ID、定妆特征（Visual Anchor）和当前坐标。
* **[REQ-02] 轴线守护逻辑：** 强制所有机位处于关系轴线的一侧，严禁出现逻辑上的“跳轴”。
* **[REQ-03] 镜头模版库：** 内置全景、过肩、特写等标准工业机位。
* **[REQ-04] 语义-参数映射：** 将形容词（如“愤怒”）映射为具体数值（如 `contrast: 0.8`, `motion: 0.7`）。

---

## 🏗️ 第二部分：设计文档 (TDD)

### 1. 系统架构图
系统采用 **Pipe-and-Filter (管道-过滤器)** 架构：
1. **Parser Filter:** 提取实体与对白。
2. **Director Filter:** 决定镜头序列与机位类型。
3. **Geometry Filter:** 计算 180° 轴线与归一化坐标。
4. **Formatter Filter:** 组装最终 JSON 协议。

### 2. 核心算法逻辑 (Copilot 开发重点)

#### A. 180° 轴线计算逻辑 (Vector-Based)
```python
# 伪代码：用于校验机位是否跳轴
def is_valid_shot(previous_cam_pos, current_cam_pos, axis_vector):
    # 核心逻辑：判断两个机位相对于轴向量的点积符号是否一致
    # 确保相机始终在轴线的同一半圆平面内
    return sign(cross_product(previous_cam_pos, axis_vector)) == sign(cross_product(current_cam_pos, axis_vector))
```

#### B. 构图坐标归一化
* 定义画布为 $1.0 \times 1.0$ 的坐标系。
* **规则：** 主体在 MCU 景别下，面部中心点应位于 $(0.33, 0.4)$ 或 $(0.66, 0.4)$，预留视线余留（Nose Room）。

### 3. 数据库/字典结构
* **Shot_Templates:** 存储不同景别对应的焦距（24mm, 35mm, 50mm, 85mm）预设。
* **Emotion_Matrix:** 情感关键词到图像参数（色彩饱和度、光影对比度）的映射表。

---

## 🛠️ 第三部分：实现方案 (Implementation)

### 1. 技术栈建议
* **语言：** Python 3.10+ (利于处理 JSON 和数学计算)。
* **逻辑大脑：** GPT-4o API (作为主要的逻辑解析器)。
* **Schema 校验：** Pydantic (确保输出的 JSON 严格符合格式)。

### 2. 关键函数模块划分
1. `script_analyzer.py`: 负责与 LLM 交互，初步拆解剧本。
2. `director_logic.py`: 核心调度逻辑，根据上下文选择 Shot Type。
3. `spatial_solver.py`: 处理所有空间几何计算和坐标映射。
4. `prompt_builder.py`: 最终字符串拼接模版。

---

## ✅ 第四部分：验收标准 (Acceptance Criteria)

为确保生成的代码符合预期，请使用以下标准进行验收：

### 1. 逻辑准确性验收 (Logic Pass)
* [ ] **轴线一致性：** 连续 5 个镜头的 JSON 输出中，`axis_side` 参数必须保持恒定（除非场景重置）。
* [ ] **视线逻辑：** 若 A 看着 B，且 A 在左侧，则 A 的 `facing` 参数必须为 `RIGHT`。

### 2. 数据完备性验收 (Data Pass)
* [ ] **必填项检查：** 每个 `Shot_Block` 必须包含 `framing`, `camera_angle`, `lighting`, `motion_instruction` 四个核心对象。
* [ ] **ID 一致性：** `CHAR_A` 的视觉特征描述在整个 JSON 序列中必须 100% 字符级匹配。

### 3. 渲染兼容性验收 (Interface Pass)
* [ ] **Prompt 质量：** 生成的 Prompt 放入开源 SD 工作流，能够直接渲染出对应景别的图像，无需人工修改词汇。
* [ ] **坐标可用性：** `pos` 坐标参数可被直接映射为 ControlNet 的坐标点。
