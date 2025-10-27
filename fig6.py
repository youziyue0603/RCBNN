import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Rectangle

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成数据
np.random.seed(42)

# 数据生成（0-0.6区间）
lower1, upper1 = 0.0, 0.6
mu1, sigma1 = 0.25, 0.08  # 均值在0.25，适中的标准差
X1 = stats.truncnorm(
    (lower1 - mu1) / sigma1, (upper1 - mu1) / sigma1, loc=mu1, scale=sigma1)
data = X1.rvs(2000)

# 缩放因子，让密度峰值达到6-7之间
scale_factor = 1.2


def create_base_plot():
    """创建基础图表"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # SCI风格配色
    hist_color = '#3B7EA1'  # 中蓝色
    line_color = '#003262'  # 深蓝色

    # 绘制直方图（density=True，然后手动缩放）
    n, bins, patches = ax.hist(data, bins=60, density=True, alpha=0.6,
                               color=hist_color, edgecolor='none')

    # 手动缩放直方图
    for patch in patches:
        patch.set_height(patch.get_height() * scale_factor)

    # 绘制核密度估计曲线并缩放
    density = stats.gaussian_kde(data)
    x_vals = np.linspace(-0.05, 0.65, 300)
    density.covariance_factor = lambda: 0.1
    density._compute_covariance()
    density_vals = density(x_vals) * scale_factor
    ax.plot(x_vals, density_vals, color=line_color, linewidth=2.5)

    # 设置图表标题和坐标轴标签
    # ax.set_title('PDF_all of Task 0', fontsize=24, pad=10)  # 字号改为24
    ax.set_ylabel('Density', fontsize=30, labelpad=10)  # 字号改为18
    ax.set_xlabel('Variance', fontsize=30, labelpad=10)  # 添加横坐标标签
    # 纵轴设置：0-7.0，间隔1.0
    ax.set_ylim(0, 7.0)
    ax.set_yticks(np.arange(0, 8.0, 1.0))

    # 横轴设置：0-0.6，间隔0.1
    ax.set_xlim(-0.05, 0.6)
    ax.set_xticks(np.arange(0, 0.7, 0.1))

    # 坐标轴样式优化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # 确保横轴0在纵轴右侧
    ax.spines['left'].set_position(('data', -0.1))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_bounds(-0.1, 0.6)

    # 刻度样式
    ax.tick_params(axis='both', length=5, width=1.5, labelsize=24)  # 字号改为24

    # 添加图例
    # ax.legend(['KDE曲线', '直方图'], loc='upper right')

    return fig, ax


# 创建并显示图表
fig, ax = create_base_plot()

# 添加三个彩色框
# 1. X=0.05处的蓝色框（0.04-0.06区间）
rect1 = Rectangle((0.04, 0), 0.06 - 0.04, 7.0,
                  linewidth=2, edgecolor='darkblue', facecolor='darkblue', alpha=0.2)
ax.add_patch(rect1)

# 2. X=0.2-0.35的绿色框
rect2 = Rectangle((0.2, 0), 0.35 - 0.2, 7.0,
                  linewidth=2, edgecolor='darkgreen', facecolor='darkgreen', alpha=0.2)
ax.add_patch(rect2)

# 3. X=0.4-0.5的橙色框
rect3 = Rectangle((0.4, 0), 0.5 - 0.4, 7.0,
                  linewidth=2, edgecolor='darkorange', facecolor='darkorange', alpha=0.2)
ax.add_patch(rect3)

plt.tight_layout()
plt.savefig('12.pdf', dpi=300, transparent=True, bbox_inches='tight')
plt.show()