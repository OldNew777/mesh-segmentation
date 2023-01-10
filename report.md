# 计算机图形学 - Mesh segmentation

陈新	计研三一	2022210877



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
- `graph.py` 为图模块，处理 mesh 对偶图。Dijkstra 算法求最短路径，Dinic 算法求最小割
- `geometry.py` 为分割模块



## 算法