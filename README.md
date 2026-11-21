# gpu-command-ring

GPU 环形命令缓冲的字节级记录与栅栏修复题。仓库开局是纯桩环境：8 个函数全部 `raise NotImplementedError`，15 个测试全部红。

## ① 题面

改成环形命令缓冲后，偶发花屏、CPU 还被等待栅栏打满：分配不按对齐，写入未对齐缓冲；跨尾不补 padding，命令被拆成两半；回收按 tick 而不是帧号，提前释放；未提交帧也回收，读到半写入数据；分帧只按条数，字节预算超限；栅栏 tick 回退不报错，等待失败；wait 用忙等，CPU 打满；对齐非 2 的幂不报错，除零；批量共用 ring 状态，帧之间串；capacity=0 不校验，除零。ring/buffer.py 是核心，tests/test_buffer.py 15个红。要修的是：沿 alignment 向上取整并在跨尾时补 padding 后整块搬移、按帧号且只回收已提交帧、按字节预算分帧、拒绝栅栏 tick 回退并用一次非阻塞检查与 1 tick 容差等待、校验对齐为 2 的幂及容量/越界/重复帧、让批量记录帧间隔离且任一帧非法时整体回滚，并对 capacity=0 抛 RingError。

## ② 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态的输出尾部必须恰好是：

```text
15 failed in <时间>s
```

即恰好 15 个 `failed`，且没有 collection error、import error 或 syntax error。环境仅使用 Python 标准库（核心实现可用 `collections`、`math`）和 pytest，离线运行，不依赖随机数；栅栏 tick 由测试显式给出。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## ④ 测试约束

`tests/test_buffer.py` 不可修改：不得删改测试、增加 skip/xfail、放宽断言，也不得通过测试侧的特殊分支绕过断言。修复应全部落在 `ring/buffer.py`，并保持 `RingError` 的异常契约以及块大小按 alignment 向上取整、帧预算按字节、栅栏 tick 单调且等待容差 1 tick 的 docstring 约定。
