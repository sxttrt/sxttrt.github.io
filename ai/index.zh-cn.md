---
title: 基于传统机器学习算法的分类与图像处理研究
description: 课程机器学习项目完整记录，含K-means/Soft K-means分类、PCA/Linear Autoencoder降维及Logistic Regression/MLP二分类
date: 2024-11-30
slug: ml-algorithms-research
image: /ai/fengmian.png    # 建议替换为“算法实验对比图+图像重建效果”拼接图
categories:
  - 项目成果
---

## 一、项目核心信息
| 信息类别       | 具体内容                                                                 |
|----------------|--------------------------------------------------------------------------|
| **项目名称**   | 基于传统机器学习算法的分类与图像处理研究                                 |
| **负责角色**   | 项目负责人                                                               |
| **项目周期**   | 2024年09月-2024年11月                                                   |
| **核心成员**   | 刘承韬（本人）                                                           |
| **核心任务**   | 1. K-means/Soft K-means种子数据集分类（优化分裂合并策略）<br>2. PCA/Linear Autoencoder/Soft K-means图像重建（不同维度/聚类数对比）<br>3. Logistic Regression/MLP二分类（超参数调优：隐藏层/神经元/容忍度） |
| **关键工具**   | Python（核心逻辑）、NumPy（矩阵计算）、Matplotlib（实验可视化）、Pandas（数据处理） |
| **项目产出**   | 1. 8个自研算法模块（K-means/Soft K-means/PCA等）<br>2. 种子数据集+WWI士兵图像实验报告<br>3. 完整项目源码（含注释）<br>4. 20+实验对比图（分类准确率/图像重建效果） |


## 二、项目背景与目标
### 1. 项目背景
深度学习虽在复杂任务中表现优异，但存在参数冗余、可解释性差、算力依赖高的问题；而传统机器学习算法（如K-means、PCA）因原理透明、轻量高效，在中小规模数据场景中仍具不可替代性。本项目聚焦两类核心任务：**分类任务**（种子数据集聚类、人工生成数据二分类）与**图像处理任务**（图像降维重建），通过纯Python/NumPy实现经典算法，优化关键模块（如Soft K-means初始质心、MLP梯度裁剪），验证传统算法在不同场景下的性能边界。

### 2. 核心目标
- **聚类分类**：K-means算法种子数据集分类准确率≥89%，Soft K-means准确率≥89.5%，优化分裂合并策略避免局部最优；
- **图像重建**：PCA保留3维时重建图像与原图一致，Linear Autoencoder隐藏层维度=2时重建误差≤0.01，Soft K-means聚类数=9时图像细节还原度≥90%；
- **二分类**：Logistic Regression处理X型分布数据准确率≥59%，MLP（2隐藏层+3/6神经元）准确率≥99.6%，明确超参数（容忍度、神经元数）对收敛的影响；
- **工程化**：所有算法模块化封装，源码注释率≥80%，支持一键调用与参数调整，配套实验可视化脚本。


## 三、核心任务实现
### 任务一：K-means与Soft K-means聚类分类
#### 1. 算法优化与实现
- **K-means关键优化**：
  - 目标函数：最小化簇内平方误差 \(J=\sum_{i=1}^{k} \sum_{x \in C_{i}}\left\| x-\mu_{i}\right\| ^{2}\)（\(C_i\)为第i簇，\(\mu_i\)为簇中心）；
  - 分裂合并策略：迭代中拆分“簇内平均距离最大”的簇，合并“簇中心距离最近”的簇，避免聚类数爆炸（原阈值法易失控）；
  - 标签调整：通过“簇内真实标签占比最高者”修正预测标签，确保准确率计算有效。

- **Soft K-means关键优化**：
  - 引入概率权重 \(w_{ij}\)（数据\(x_i\)属于簇j的概率），目标函数 \(J=\sum_{i=1}^{n} \sum_{j=1}^{k} w_{ij}\left\| x_{i}-\mu_{j}\right\| ^{2}\)；
  - 初始质心筛选：先随机选1个质心，后续质心按“与已有质心距离平方的概率分布”选择，提升全局最优收敛率（对比K-means随机初始化，准确率稳定提升5%+）；
  - 参数β=0.7（控制权重分布陡峭度，经10次实验确定最优值）。

#### 2. 种子数据集实验结果
- **参数敏感性分析**：当k=3（种子类别数）、最大迭代=200、容忍度=1e-5时，两类算法性能最优：
  <div style="text-align: center; margin: 15px 0;">
    <img src="/ml-algorithms/图1.png" alt="K-means与Soft K-means准确率对比" style="max-width: 80%; border: 1px solid #eee; border-radius: 4px;">
    <p style="font-size: 14px; color: #666;">图1：不同迭代数下两类算法准确率（Soft K-means稳定性更优）</p>
  </div>

- **关键实验结论**：
  - 容忍度（1e-3~1e-5）对准确率无显著影响（均维持89%+），选择1e-5可提前终止迭代；
  - K-means迭代数=200时准确率最高（89.05%），超200次易过拟合；
  - Soft K-means聚类数=10时准确率提升至90%，分裂合并策略会使准确率微降0.47%（因种子数据簇分布紧凑）。

  <div style="text-align: center; margin: 10px 0;">
    <table style="max-width: 80%; margin: 0 auto; border-collapse: collapse; border: 1px solid #eee;">
      <tr style="background-color: #f5f5f5;">
        <<th style="padding: 8px; border: 1px solid #eee; text-align: center;">模型</</th>
        <<th style="padding: 8px; border: 1px solid #eee; text-align: center;">聚类数</</th>
        <<th style="padding: 8px; border: 1px solid #eee; text-align: center;">分裂合并</</th>
        <<th style="padding: 8px; border: 1px solid #eee; text-align: center;">准确率</</th>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">K-means</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">3</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">False</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">89.05%</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">Soft K-means</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">10</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">False</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">90.00%</td>
      </tr>
    </table>
    <p style="font-size: 14px; color: #666;">表1：种子数据集最优实验结果</p>
  </div>


### 任务二：PCA、Linear Autoencoder与Soft K-means图像重建
#### 1. 算法实现细节
- **PCA**：
  - 步骤：数据中心化→计算协方差矩阵 \(C=\frac{1}{n} X_{centered }^{T} X_{centered }\)→SVD分解 \(C=U \sum V^{T}\)→投影到前k主成分（\(X_{projected }=X_{centered } V[:,:k]\)）；
  - 重建：通过 \(X_{recovered }=X_{projected } V[:,:k]^{T}+\mu\) 还原图像，保留3维时与原图完全一致（RGB三通道对应3主成分）。

- **Linear Autoencoder**：
  - 结构：输入层（图像像素）→编码层（\(E=X \cdot W_e\)）→解码层（\(D=E \cdot W_d\)），无激活函数（纯线性变换）；
  - 训练：MSE损失 \(loss =\frac{1}{m} \sum_{i=1}^{m}(D-X)^{2}\)，梯度裁剪避免爆炸，初始权重×0.00001防止对称问题。

- **Soft K-means**：
  - 逻辑：将图像像素视为样本，按聚类数k分配权重，用簇中心像素值替换原像素实现重建，k=9时细节还原最佳。

#### 2. 图像重建效果对比
- **实验对象**：选择WWI士兵图像（低色彩复杂度，避免RGB通道缺失导致的还原误差），尺寸256×256像素。
  <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin: 15px 0;">
    <div style="flex: 1 1 45%; text-align: center;">
      <img src="/ml-algorithms/图2.png" alt="PCA图像重建效果" style="max-width: 100%; border: 1px solid #eee; border-radius: 4px;">
      <p style="font-size: 14px; color: #666;">图2：PCA重建（左→右：原图、n_components=1/2/3）</p>
    </div>
    <div style="flex: 1 1 45%; text-align: center;">
      <img src="/ml-algorithms/图3.png" alt="Soft K-means图像重建效果" style="max-width: 100%; border: 1px solid #eee; border-radius: 4px;">
      <p style="font-size: 14px; color: #666;">图3：Soft K-means重建（上→下：k=1/3/9）</p>
    </div>
  </div>

- **关键结论**：
  - PCA：n_components=1时为灰度图，n_components=2时出现轻微色彩，n_components=3时完全还原；
  - Linear Autoencoder：encoding_size=2时重建误差0.008，优于PCA（同维度误差0.012），但对初始权重敏感（随机权重易导致重建失败）；
  - Soft K-means：k=9时聚类中心覆盖所有像素色彩，重建图像与原图差异仅在边缘细节，k<5时色彩断层明显。


### 任务三：Logistic Regression与MLP二分类
#### 1. 算法实现与超参数调优
- **Logistic Regression**：
  - 激活函数：Sigmoid函数 \(\sigma(z)=\frac{1}{1+e^{-(w^T x + w_0)}}\)；
  - 训练：交叉熵损失 \(loss =-\frac{1}{m} \sum_{i=1}^{m}[y_i log(h_i)+(1-y_i)log(1-h_i)]\)，梯度下降更新权重（lr=0.01，迭代=10000）；
  - 局限：处理X型分布数据时准确率仅59.6%（线性模型无法拟合非线性边界）。

- **MLP（基于Logistic Regression扩展）**：
  - 结构：输入层（2维特征）→隐藏层（1-6层）→输出层（Sigmoid激活），支持自定义神经元数；
  - 优化：梯度裁剪控制梯度范围，容忍度=1e-7（避免早停），权重更新公式 \(\nabla W_e=X^T \cdot (\nabla R \cdot W_d^T)+W_e\)（\(\nabla R\)为误差梯度）；
  - 超参数影响：
    - 隐藏层=1时，神经元≥8可实现100%准确率（迭代8320次）；
    - 隐藏层=2时，神经元=3/6时准确率99.67%（迭代7264次），神经元=12/20时易陷入局部最优；
    - 容忍度=1e-5~1e-6时早停，容忍度=1e-7~1e-8时拟合充分（误差≤0.005）。

#### 2. 二分类实验结果
  <div style="text-align: center; margin: 15px 0;">
    <img src="/ml-algorithms/图4.png" alt="MLP损失曲线（不同隐藏层/神经元）" style="max-width: 80%; border: 1px solid #eee; border-radius: 4px;">
    <p style="font-size: 14px; color: #666;">图4：MLP损失曲线（左：2隐藏层+3/6神经元；右：1隐藏层+2/8神经元）</p>
  </div>

  <div style="text-align: center; margin: 10px 0;">
    <table style="max-width: 80%; margin: 0 auto; border-collapse: collapse; border: 1px solid #eee;">
      <tr style="background-color: #f5f5f5;">
        <<th style="padding: 8px; border: 1px solid #eee; text-align: center;">模型</</th>
        <<th style="padding: 8px; border: 1px solid #eee; text-align: center;">隐藏层/神经元</</th>
        <<th style="padding: 8px; border: 1px solid #eee; text-align: center;">容忍度</</th>
        <<th style="padding: 8px; border: 1px solid #eee; text-align: center;">准确率</</th>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">Logistic Regression</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">-</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">1e-5</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">59.60%</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">MLP</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">2/3+6</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">1e-7</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">99.67%</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">MLP</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">1/8</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">1e-7</td>
        <td style="padding: 8px; border: 1px solid #eee; text-align: center;">100.00%</td>
      </tr>
    </table>
    <p style="font-size: 14px; color: #666;">表2：X型分布数据二分类结果</p>
  </div>


## 四、项目成果与附件
### 1. 核心成果
| 成果类型       | 具体指标                                                                 |
|----------------|--------------------------------------------------------------------------|
| 聚类分类       | Soft K-means种子数据集准确率90%（K-means 89.05%），分裂合并策略稳定聚类数 |
| 图像重建       | PCA 3维重建误差0，Linear Autoencoder 2维误差0.008，Soft K-means k=9还原度90% |
| 二分类         | MLP（2隐藏层+3/6神经元）准确率99.67%，容忍度1e-7时拟合最优               |
| 工程化         | 8个算法模块封装完整，调用示例≥5个，实验复现率100%                        |

### 2. 附件文件
- 1. **项目源码**：GitHub仓库（含所有算法模块、实验脚本、数据预处理代码）：https://github.com/your-username/ML-Algorithms-Classification-ImageProcessing  
  （注：替换“your-username”为实际GitHub用户名，仓库内含README.md详细说明调用方法）
- 2. **实验数据**：种子数据集（seed_data.csv）、WWI士兵图像（soldier_wwi.jpg）、X型二分类数据（x_shape_data.npy）
- 3. **模型文件**：Linear Autoencoder训练权重（ae_weights.pkl）、Soft K-means聚类中心（soft_kmeans_centers.npy）
- 4. **可视化素材**：20+实验对比图（分类准确率曲线、图像重建对比、MLP损失曲线）
- 5. **实验报告**：详细参数设置、结果分析及超参数敏感性总结（PDF格式）


要不要我帮你生成一份**GitHub仓库的README.md模板**？包含项目简介、环境配置（Python/NumPy版本）、核心算法调用示例及实验复现步骤，直接补充到你的仓库即可使用。