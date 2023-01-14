# 计算机图形学 - Mesh segmentation

陈新	计研三一	2022210877



**注：检查时完成 k 路分解，新增层次化 k 路分解**



## 代码结构

> | -- data
>
> | -- mylogger
>
> | -- outputs
>
> | -- func.py
>
> ​       geometry.py
>
> ​       graph.py
>
> ​       mesh_segmentation.py



- `data` 为 `ply` 数据文件夹
- `mylogger` 为日志模块
- `outputs` 为分割输出文件夹
- `mesh_segmentation.py` 为程序主入口，处理命令行参数、完成初始化工作
- `graph.py` 为图模块，读取 **ply**（plyfile库）**/obj**（手写 parser） 文件，处理 mesh 对偶图，Dijkstra 算法求最短路径，Dinic 算法求最小割
- `geometry.py` 为几何分割主要的算法性代码模块



此外附带一个我曾经写的 C++ 的 OpenGL 渲染器，OpenGLRenderer，可以用于流水线式地渲染几何分割脚本生成的场景文件



## 运行方式

### 参数说明

- `-f`, `--filename`： 待分割的 ply 模型文件路径，默认为 `./data/horse.ply`
- `-k`, `--k_max`：分类的最大 k 值，默认为 20
- `-s`, `--seed`：随机数种子，默认为 3984572



### 运行脚本

在 `mesh-segmentation` 文件夹下

```
python mesh_segmentation.py
```

则会生成 obj/ply 格式的模型文件，以及 OpenGLRenderer 格式的场景文件



在 `OpenGLRenderer` 文件夹下 cmake 配置完成后

```
./build/bin/opengl-render-cli.exe -s #路径#/opengl-render/scene.json
```

则可以通过渲染器查看分割结果



## 算法/复杂度分析

完成 k 路层次化分割**（注：检查时完成 k 路分解，新增层次化 k 路分解）**



1. 对于每一层层次化迭代：
    1. 读取上一轮迭代的结果作为输入模型
    2. 生成 mesh 的带权对偶图，并计算两个面片间的最短距离。使用 `PriorityQueue` 的 `Dijkstra` 算法，F个出发点各运行一次，因此复杂度 $o(F^2 \log F)$
       1. 其中对于相邻面片 $f_{i}, f_{j}$
          1. 角距离：$\text{Ang\_Dist}(\alpha_{ij}) = \eta (1 - \cos \alpha_{ij})$ 凸的情况下 $\eta < 1$ （代码中取0.2），凹的情况下 $\eta = 1$
          2. 测地线距离：$\text{Geod\_Dict}$ 面片质心之间经过模型表面的最短连线
          3. 权值：$\text{Weight}(f_{i}, f_{j}) = \delta \frac{\text{Geod\_Dict}(f_{i}, f_{j})}{\text{avg}(\text{Geod\_Dict})} + (1 - \delta) \frac{\text{Ang\_Dict}(\alpha_{ij})}{\text{avg}(\text{Ang\_Dict})}$
    3. 一个一个地插入新的 REP 直至 k 个，计算 $Gk$ 的值，并根据其一阶梯度确定最佳 k 值选项。复杂度 $o(k^2 F)$
    4. 迭代地执行至多 $\text{max\_iter}$ 次
       1. 确定 k 个 $p$ 值最大的 REP
       2. 为每个面片 $f_i$ 计算属于每一片 $S_l$ 的概率
       2. 若 REP 有更新，则继续迭代
       2. 复杂度 $o(kF + kF^2 + kF^2) = o(kF^2)$
    5. 对比较确定的面片，直接进行划分；对模糊区域使用 `Dinic` 最小割算法来划分。每次使用 `Dinic` 算法求最小割需要 $o(F \times \frac{3}{2} F) = o(F^2)$，因此该部分复杂度 $o(kF + k^2 \times (F + F^2)) = o(k^2 F^2)$
2. 最终合并为结果
   



故总复杂度为 $o((\log F + \text{max\_iter} + k^2) \times F^2)$



## 实验结果

Windows 11，i9-9900KF

在 `horse` 样例下（2400个顶点，4796个面片）

| 层次 | Result                                                     |
| ---- | ---------------------------------------------------------- |
| 1    | <img src="pictures/k=2.png" alt="k=2" style="zoom:50%;" /> |
| 2    | <img src="pictures/k=4.png" alt="k=4" style="zoom:50%;" /> |

