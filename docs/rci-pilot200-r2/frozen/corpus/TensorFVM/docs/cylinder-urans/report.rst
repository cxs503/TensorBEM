Re=3900 外流圆柱 SA-URANS 诊断
================================

本算例为直径 D=1 的外流圆柱，Re_D=3900；计算域为 20D × 12D，圆心在
(5D, 6D)，上下外边界为均匀来流远场而非通道壁面。使用 O 型贴体网格、
原始完全湍流 Spalart--Allmaras 闭合、隐式 Euler 时间推进和 SIMPLE 内迭代。
在初始速度中加入固定的反对称微扰，以避免精确对称初值掩盖尾迹不稳定性。

验收不是将二维 URANS 宣称为 Re=3900 的充分模型：该流动存在三维湍流尾迹，
文献通常以 LES/DES 或三维计算验证。这里要求同时解析出非零升力波动和
0.1 < St < 0.3；若未满足，程序明确报告未通过，而不是用稳态对称解冒充
涡脱落。参考时均 Cd=1.12、St=0.20，来自 Parnaudeau et al. (2008), Experimental and numerical studies of the flow over a circular cylinder at Reynolds number 3900。
仅在已解析周期信号、每个物理时间步内残差收敛且 Cd/St 分别严格小于
25% / 20% 误差时通过。

结果
----

* 物理步数：300
* 后处理样本：200
* 平均 Cd：1.158879（参考 1.12）
* Cl RMS：1.070505e-05
* St：0.1（参考 0.2）
* 是否解析周期升力：False
* 是否通过：False

复现::

    python -m tensorfvm.benchmark_cylinder_urans --output results/cylinder-urans

输出中的 ``forces.json`` 是逐物理时间步的 Cd/Cl 历史；它是 St 计算的原始数据。
