import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.colors as mcolors

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 创建图形布局：上面3个子图，下面2个子图
fig = plt.figure(figsize=(24, 16))

# 定义颜色方案（二分类只需要两种颜色）
colors = ['#4ECDC4', '#FF6B6B']  # 蓝色和橙色，与图片一致


def generate_tsne_like_data(n_samples=800, centers_pos=None, cluster_std=0.8, shape_params=None, random_state=None):
    """生成类似T-SNE的二分类数据，根据红框位置定制分布"""
    if random_state is not None:
        np.random.seed(random_state)

    if centers_pos is None:
        centers_pos = [[-2, -2], [2, 2]]

    if shape_params is None:
        shape_params = [{'angle': 0, 'stretch_x': 1, 'stretch_y': 1},
                        {'angle': 0, 'stretch_x': 1, 'stretch_y': 1}]

    # 生成基础聚类数据
    X, y = make_blobs(n_samples=n_samples, centers=centers_pos,
                      cluster_std=cluster_std, random_state=random_state)

    # 对每个簇分别进行变形以匹配红框形状
    for i, label in enumerate(np.unique(y)):
        mask = (y == label)
        cluster_points = X[mask]

        # 获取该簇的形状参数
        params = shape_params[i]
        angle = params['angle']
        stretch_x = params['stretch_x']
        stretch_y = params['stretch_y']

        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # 中心化
        center = np.mean(cluster_points, axis=0)
        centered = cluster_points - center

        # 拉伸和旋转
        stretch_matrix = np.array([[stretch_x, 0], [0, stretch_y]])
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        transformed = centered @ stretch_matrix @ rotation_matrix
        X[mask] = transformed + center

    return X, y


# 重新设计配置，确保旋转后簇之间有足够距离，不会重叠
fig_configs = [
    # Fig1: 左上角蓝色簇，右下角橙色簇，增大距离确保分离
    {
        'centers': [[-1.8, 1.2], [1.8, -1.2]],
        'shapes': [
            {'angle': np.pi / 4, 'stretch_x': 1.2, 'stretch_y': 0.7},  # 蓝色簇，旋转45度
            {'angle': -np.pi / 6, 'stretch_x': 1.1, 'stretch_y': 0.8}  # 橙色簇，旋转-30度
        ]
    },
    # Fig2: 上方橙色簇，下方蓝色簇，保持足够垂直距离
    {
        'centers': [[0, 1.8], [0, -1.8]],
        'shapes': [
            {'angle': np.pi / 6, 'stretch_x': 1.4, 'stretch_y': 0.6},  # 蓝色簇，旋转30度
            {'angle': -np.pi / 8, 'stretch_x': 1.6, 'stretch_y': 0.5}  # 橙色簇，旋转-22.5度
        ]
    },
    # Fig3: 左上角橙色簇，右下角蓝色簇，对角分离
    {
        'centers': [[-1.5, 1.3], [1.7, -1.1]],
        'shapes': [
            {'angle': np.pi / 3, 'stretch_x': 1.0, 'stretch_y': 0.8},  # 蓝色簇，旋转60度
            {'angle': -np.pi / 4, 'stretch_x': 1.3, 'stretch_y': 0.6}  # 橙色簇，旋转-45度
        ]
    },
    # Fig4: 上方蓝色簇，下方橙色簇，增大垂直距离
    {
        'centers': [[0, 1.6], [0, -1.6]],
        'shapes': [
            {'angle': np.pi / 5, 'stretch_x': 1.5, 'stretch_y': 0.6},  # 蓝色簇，旋转36度
            {'angle': -np.pi / 7, 'stretch_x': 1.7, 'stretch_y': 0.5}  # 橙色簇，旋转-25.7度
        ]
    },
    # Fig5: 左边橙色簇，右边蓝色簇，增大水平距离
    {
        'centers': [[-2.0, 0], [2.0, 0]],
        'shapes': [
            {'angle': np.pi / 6, 'stretch_x': 0.7, 'stretch_y': 1.2},  # 蓝色簇，旋转30度
            {'angle': -np.pi / 6, 'stretch_x': 0.6, 'stretch_y': 1.4}  # 橙色簇，旋转-30度
        ]
    }
]

# 增大簇标准差，让点分布更离散
fixed_cluster_std = 0.8  # 从0.5增加到0.8

# 第一行的3个子图
for i in range(3):
    ax = plt.subplot(2, 3, i + 1)

    config = fig_configs[i]
    X, y = generate_tsne_like_data(n_samples=600,
                                   centers_pos=config['centers'],
                                   cluster_std=fixed_cluster_std,
                                   shape_params=config['shapes'],
                                   random_state=42 + i * 17)

    # 绘制散点图（二分类）
    unique_labels = np.unique(y)
    for j, label in enumerate(unique_labels):
        mask = (y == label)
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[j],
                   s=25, alpha=0.8, edgecolors='none')

    # 设置标题和标签
    # ax.set_title(f'T-SNE fig{i + 1}', fontsize=24)
    # ax.set_xlabel('t-SNE 1', fontsize=18)
    # ax.set_ylabel('t-SNE 2', fontsize=18)

    # 设置刻度字号
    ax.tick_params(labelsize=24)

    # 设置网格
    ax.grid(True, alpha=0.3)

    # 去掉上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置坐标轴范围
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)

# 获取第一行子图的实际位置和大小，以确保第二行子图大小一致
first_row_ax = fig.get_axes()[0]
pos = first_row_ax.get_position()
subplot_width = pos.width
subplot_height = pos.height

# 第二行的2个子图 - 使用add_axes精确控制位置，确保大小一致
bottom_row_y = 0.05  # 底部行的y位置

# 计算居中位置：总宽度是1，两个图加间距的总宽度
total_width = 2 * subplot_width + 0.05  # 两个图宽度 + 间距
start_x = (1 - total_width) / 2  # 居中起始位置

# 第一个图
left1 = start_x
ax4 = fig.add_axes([left1, bottom_row_y, subplot_width, subplot_height])

# 第二个图
left2 = left1 + subplot_width + 0.05  # 添加间距
ax5 = fig.add_axes([left2, bottom_row_y, subplot_width, subplot_height])

axes_bottom = [ax4, ax5]

for i, ax in enumerate(axes_bottom):
    config = fig_configs[i + 3]
    X, y = generate_tsne_like_data(n_samples=600,
                                   centers_pos=config['centers'],
                                   cluster_std=fixed_cluster_std,
                                   shape_params=config['shapes'],
                                   random_state=42 + (i + 3) * 17)

    # 绘制散点图（二分类）
    unique_labels = np.unique(y)
    for j, label in enumerate(unique_labels):
        mask = (y == label)
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[j],
                   s=25, alpha=0.8, edgecolors='none')

    # 设置标题和标签
    # ax.set_title(f'T-SNE fig{i + 4}', fontsize=24)
    # ax.set_xlabel('t-SNE 1', fontsize=18)
    # ax.set_ylabel('t-SNE 2', fontsize=18)

    # 设置刻度字号
    ax.tick_params(labelsize=24)

    # 设置网格
    ax.grid(True, alpha=0.3)

    # 去掉上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置坐标轴范围
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)

# 保存图片
plt.savefig('1.pdf', dpi=600, transparent=True, bbox_inches='tight')
plt.show()