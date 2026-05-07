from graphviz import Digraph

# 创建流程图对象
dot = Digraph('DN_Iteration', comment='D-N Iteration Flowchart')
dot.attr(rankdir='TB', size='8,10')
dot.attr('node', fontname='SimSun', shape='rectangle', style='filled', fillcolor='white') # 使用宋体

# 定义节点样式
dot.node('Start', '开始: 初始化界面参数 Γ\n及初始值 Γ⁰', shape='ellipse', fillcolor='#F9F9F9')
dot.node('Init', '设定外部迭代步数 k = 0')

# 子图：循环部分
with dot.subgraph(name='cluster_iteration') as c:
    c.attr(label='第 k+1 次全局迭代', style='dashed', color='blue')
    c.node('Step1', '步骤 1: 子域 Ω₁ Dirichlet 求解\n(以 Γᵏ 为 Dirichlet 边界条件)')
    c.node('AD', '自动微分 (AD) 提取\n界面法向梯度 ∇u₁·n', shape='parallelogram', fillcolor='#E1F5FE')
    c.node('Step2', '步骤 2: 子域 Ω₂ Neumann 求解\n(将梯度作为通量条件)')
    c.node('Step3', '步骤 3: 界面结果融合与松弛更新\nΓᵏ⁺¹ = θ·u₂|_Γ + (1-θ)·Γᵏ', fillcolor='#FFF9C4')

# 判定与结束
dot.node('Judge', '收敛判断:\n||Γᵏ⁺¹ - Γᵏ|| < τ?', shape='diamond', fillcolor='#FFF9C4')
dot.node('Update', '更新步数: k = k + 1')
dot.node('End', '结束: 输出耦合物理场', shape='ellipse', fillcolor='#F9F9F9')

# 连接线
dot.edge('Start', 'Init')
dot.edge('Init', 'Step1')
dot.edge('Step1', 'AD')
dot.edge('AD', 'Step2')
dot.edge('Step2', 'Step3')
dot.edge('Step3', 'Judge')
dot.edge('Judge', 'Update', label='否')
dot.edge('Update', 'Step1')
dot.edge('Judge', 'End', label='是')

# 保存图片
# 如果你安装了 Graphviz 软件，它可以直接渲染出图片
try:
    dot.render('DN_Iteration_Flowchart', format='png', view=True)
    print("流程图已生成并保存为 DN_Iteration_Flowchart.png")
except:
    print("代码已生成，但检测到系统中未配置 Graphviz 执行环境。")
    print("你可以直接将以下内容粘贴到网站：")
    print(dot.source)