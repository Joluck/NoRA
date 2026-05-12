# 基于权重⼏何的 RLVR 最速下降优化器：完整推导计划
从 Three-Gate Theory 的物理直觉出发，推导⼀个 GPU 友好的 RLVR-native 优化器。
# 第零步：为什么不是 Hessian
在经典优化中，最速下降的度量选择是 loss 的 Hessian（曲率矩阵），对应 Newton 步。但在RLVR 中，这条路⾛不通：
问题 1：RLVR 没有稳定的 Hessian。 策略梯度的 surrogate loss $L_{PG}(\theta) = -\mathbb{E}[A^\perp \log \pi_\theta(y|x)]$ 每步采样新 rollout 就换⼀个 loss landscape。
Hessian 是某个固定函数在某⼀点的⼆阶局部信息，但 RLVR 的 loss 本身每步都在变⸺给⼀个不断变形的曲⾯计算曲率没有意义。
问题 2：即使没有 KL penalty，off-principal ⾏为仍然存在。 论⽂⽤ DAPO（ $\$ 10 e t a=0$ ，完全没有 KL 惩罚项）做实验，off-principal 更新依然存在（Figure 15 显示即使没有 KL losspenalty，token-wise KL 仍然稳步增⻓但被约束）。这说明约束不来⾃ loss 的曲率，⽽来⾃$\$ 103$ 本身的参数化结构。
问题 3：Newton 步在 RL 中没有意义。 $\$ 106$ 在凸优化中跳到极⼩值，但RLVR 的 loss ⾮凸、⾮平稳，Newton ⽅向可能指向鞍点甚⾄极⼤值。
结论：约束应该来⾃权重的谱结构，不是 loss 的曲率。
# 第⼀步：正确的物理直觉
# 1.1 Three-Gate Theory 告诉我们什么
Gate I（KL Anchor）：每步策略变化 $D_{KL}(\pi_{\theta^+} | \pi_\theta) $\$ 1$ 被限制
Gate II（Model Geometry）： $\$ 0$ 的奇异值结构决定了不同⽅向 $\Delta W$ 对模型⾏为的影响⼤⼩
Gate III（Precision）：bfloat16 过滤微更新，是表观现象，⾮本质
Gate II 是关键。Wedin 的 $\sin\Theta$ 定理：
```txt
$$ \sin\Thetaeta_k \leq \frac{1}{2} \{ |Delta_W|2 \} \{ |gamma_k \}, |quad|gamma_k = |sigma_k(W_0) - sigma_{k+1}(W_0) $ 
```
含义：
⼤ gap ⽅向（principal）：⼦空间旋转被强约束，动⼀点就”代价”很⼤
⼩ gap ⽅向（off-principal）：约束弱，可以动很多
# 1.2 正确的最速下降⽬标
不是 Hessian 度量下的最速下降，⽽是 $\$ 103$ -⼏何度量下的最速下降：
```txt
$$ \min_{\{Delta W\}}; \text{Tr}(G^{\wedge}\top \Delta Delta W) \quad quad \text{s.t.} \quad quad |Delta W|_{\{W\text{text{-geo}}\}}^{\wedge 2} \quad \leq \text{eta}^{\wedge 2} $ 
```
其中 ${ } ^ { \mathfrak { n } } \$ \mathsf { W } \$ .$ -⼏何度量”的含义是：沿不同奇异⽅向⾛⼀步的”代价”不同⸺principal ⽅向贵，off-principal ⽅向便宜。
# 第⼆步：在 SVD 基下建⽴加权范数
# 2.1 换坐标系
```txt
$W = U\Sigma V^{\wedge}\top$, 把梯度投影到 $W$ 的奇异基:
```
```perl
$$ \tilde{tilde} \{G\} = U^{\wedge}\top G V $$$ 
```
$\tilde{G}_{ij} $\$ 9$ 的物理含义：梯度在第 $\$ 1$ 个左奇异⽅向、第 $\$ 1\$ 8$ 个右奇异⽅向上的分量。
# 2.2 定义加权范数
给每个⽅向⼀个”价格” $\$ 0$ ：
```txt
$$ |\Delta tla W|c^2 = |sum{i,j} c_i \cdot{tilde{\Delta tla}}_{}_{\{ij\}}^2 $$ 
```
$\$ 0$ ⼤ $=$ 第 $\$ 1$ 个奇异⽅向（对应 $\sigma_i$）“贵”。
# 2.3 最速下降问题
```latex
$$ \min_{\{tilde{\Delta t}a\}} \sum_{\{i j\}} \tilde{tilde}{G}\{i j\}, |tilde{|Delta}\{i j\} \quad \text{quad} \text{text{s.t.}} \quad \text{quad} \text{sum}_{\{i j\}} c_{i} \tilde{tilde{\Delta t}a}\{i j\}^{\wedge} 2 \text{leq} \text{eta}^{\wedge} 2 $ 
```
# 第三步：求解⸺拉格朗⽇乘⼦法
# 3.1 KKT 条件
对每个分量 $\$ ( i, j)\$ 5$ 独⽴：
$$
\$ \$\$\tilde{t}ilde\{G\}\{ij\} + 2|l a m b d a, c\_i,\text{|tilde}{\{\text{D e l t a}\}}\{i j\} = 0\$
$$
解出：
$$
\$ \$\tilde{tilde}\{\Delta Delta\} \{ij\} = -|frac\{|tilde{G}\{ij\} \} \{2\backslash lambda,c\_ i\} \$
$$
$2\lambda$ 是全局缩放因⼦，被学习率吸收。本质上：
$$
\$ \$\boxed{\tilde{d}\{\Delta t l e\{\Delta d e t a\}i j\}}|p r o p t o -|f r a c{|t i l d e G}{i j}\{\{c_{-}i\})
$$
每个⽅向的更新 $=$ 梯度 $\div$ 该⽅向的价格。
# 3.2 变回原坐标
$$
\$ \$\backslash D e l t a W = U, \tilde {t} i l d e \{\Delta d e l t a \}, V ^ {\wedge} \backslash t o p \$ \$\
$$
由于 $\$ 0$ 只依赖⾏索引 $\$ 19$ （左奇异⽅向），可以写成矩阵形式：
$$
\$ \$\backslash D e l t a W = - U, C ^ {\wedge} \{- 1 \}, U ^ {\wedge} \backslash t o p, G \$
$$
其中 $\$ 0=$ \text{diag}(c_1, c_2, \ldots)$。
# 3.3 数值例⼦
取 $\$ 123,456$ ，设 $\$ 123,456$ 
<table><tr><td>方向</td><td>\$\\sigma_i\$</td><td>梯度 \$\\tilde{G}\}_i\$</td><td>价格 \$c_i= \\sigma_i^2\$</td><td>更新 \$- \\tilde{G}\_i/c_i\$</td></tr><tr><td>principal</td><td>10</td><td>5</td><td>100</td><td>-0.05</td></tr><tr><td>off-principal</td><td>1</td><td>3</td><td>1</td><td>-3.0</td></tr></table>
更新在 off-principal ⽅向⼤了 60 倍⸺和论⽂观察到的 RLVR ⾏为⼀致。
# 第四步：消除 SVD⸺转化为矩阵运算
# 4.1 关键恒等式
选 $c_i = \sigma_i^2 + \epsilon$（价格 $=$ 奇异值平⽅ $^ +$ 正则项），注意到：
$$
\begin{array}{l} \\ \text{\$\$\$\$\^{\wedge}\top = U\Sigma m^{\wedge}2U^{\wedge}\top}\\ \end{array}
$$
因此：
$$
\mathbb {S} \mathbb {S} \mathbb {U}, \backslash \text {t e x t} \{\text {d i a g} \} (\backslash \text {s i g m a} _ {\mathrm {i}} \wedge 2 + \backslash \text {e p s i l o n}) ^ {\wedge} \{- 1 \}, \mathbb {U} ^ {\wedge} \backslash \text {t o p} = (\mathbb {W} \mathbb {W} ^ {\wedge} \backslash \text {t o p} + \backslash \text {e p s i l o n I}) ^ {\wedge} \{- 1 \} \mathbb {S} \mathbb {S}
$$
# 4.2 ⽆ SVD 的更新公式
$$
\$ \$\backslash b o x e d\{\backslash D e l t a W = -\backslash e t a; (W W ^ {\wedge} \backslash t o p + \backslash e p s i l o n I) ^ {\wedge} \{- 1 \}; G \} \$\$
$$
SVD 完全消失了！ 不需要计算 $U, V, \Sigma$，只需要解⼀个线性⽅程组：
$$
\$ \$\$(W W ^ {\wedge} \backslash t o p + \backslash e p s i l o n I); \backslash D e l t a W = - \backslash e t a; G \$\$\$
$$
# 4.3 物理含义
$WW^\top$ 的特征值是 $\$ 1$ \sigma_ $\mathsf { i } \land 2 \$ 8$ ⸺编码了 $\$ 0$ 的谱结构
· $\$ +$ \epsilon $| \$ 1$ 是正则化，防⽌⼩奇异值⽅向爆炸
· $\$ 100$ ⾃动实现”principal ⽅向缩⼩，off-principal ⽅向放⼤”
# 第五步：GPU 友好的计算⽅法
# 5.1 ⽅法 A：共轭梯度法（CG）（推荐）
解 $\$ 0$ $\boldsymbol { \mathbf { \rho } } _ { \mathsf { X } } = \mathsf { g } \boldsymbol { \Phi }$ 的每⼀列。
每次 CG 迭代只需要：
1. 计算 $W^\top $\nu \$ 1$ ：⼀次矩阵乘法
2. 计算 $\$ 123,456,7$ ：⼜⼀次矩阵乘法
3. 加 $\epsilon $\nu \up$ 1$ ：向量加法
不需要显式形成 $WW^\top$（这是 $\$ 1$ \times $\mathsf { n } \$ 9$ 的⼤矩阵）。
CG 的收敛速度取决于条件数 $\kappa = (\sigma_1^2 + \epsilon)/(\sigma_r^2 + \epsilon)$。$\epsilon$ 越⼤，条件数越好，收敛越快。通常 5-10 次迭代就够了。
总计算量：每步约 10-20 次矩阵乘法，和 Muon 的 Newton-Schulz 迭代（5 次三阶多项式 $= 1 0$ 次矩阵乘法）在同⼀量级。
# 5.2 ⽅法 B：Cholesky 分解
如果 $\$ 123$ （⾏维度）不太⼤：
1. 形成 $A = WW^\top $^ +$ \epsilon I$（$n \times n$）
2. Cholesky 分解 $\$ 4$ 
3. 两次三⻆求解： $\$ 1 y =6\$ 8$ ，$L^\top \Delta W = y$
cuSOLVER 直接⽀持。适合 $\$ 1$ $\$ 123,456$ 的场景。
# 5.3 ⽅法 C：Neumann 级数近似
```latex
$$ (WW^{\top} + \epsilonpsilonilon 1)^{\wedge}\{-1\} = \{\text{frac}\{1\}\{\epsilonilon 1\} \left(1 + \text{frac}\{WW^{\wedge}\top\}\right) \{\epsilonilon 1\} \text{right})^{\wedge}\{-1\} \text{approx} \{\text{frac}\{1\}\{\epsilonilon 1\} \text{sum}_{-}\{k=0\}^{\wedge}\{K\}(-1)^{\wedge}k \left(\text{frac}\{WW^{\wedge}\top\}\right) \{\epsilonilon 1\} \text{right})^{\wedge}k $ 
```
每⼀项都是矩阵乘法链。但只在 $|WW^\top|/\epsilon $< 1 \$ 1$ 时收敛，即 $\epsilon >\sigma_ $1 \sim 2 \$ 1$ ，条件过于严格，⼀般不实⽤。
# 5.4 计算量对⽐
<table><tr><td>方法</td><td>每步操作</td><td>适用条件</td><td>额外内存</td></tr><tr><td>精确 SVD</td><td>$O(nm\min(n,m))$</td><td>小矩阵</td><td>$U, V$</td></tr><tr><td>CG(推荐)</td><td>$K$ 次 $(W(W^{\wedge}\top v))$, $K\approx approx 5\text{ text{-}}10$</td><td>通用</td><td>几个向量</td></tr><tr><td>Cholesky</td><td>$O(n^3) + O(n^2 m)$</td><td>$n$ 不太大</td><td>$LL^{\wedge}\top$</td></tr><tr><td>Neumann</td><td>$k$ 次矩阵乘法</td><td>\$εpsilon &gt; \sigma_{\text{sigma\_1}}^2 \Sigma</td><td>无</td></tr></table>
# 第六步：$\epsilon$ 的选择
$\epsilon$ 控制”off-principal 放⼤”的激进程度：
$\epsilon$ 很⼤（$\epsilon \gg \sigma_1^2$）：$(WW^\top $^ +$ \epsilon I)^{-1} \approx\frac{1}{\epsilon}I$，退化为 SGD
$\epsilon$ 很⼩（$\epsilon \ll \sigma_ $\mathsf { r } ^ { \wedge } 2 \$ 1$ ）：⼩奇异值⽅向被极度放⼤，噪声⻛险
· $\$ 1$ \epsilon \sim \sigma_{\text{median}} $\land 2 \$ 1$ ：中间状态，温和地偏向 off-principal
建议：$\epsilon $=$ \alpha \cdot \text{mean}(\sigma_i^2) $=$ \alpha \cdot |W|_F^2 /$\mathsf { l m i n } ( \mathsf { n } , \mathsf { m } ) \$ $ ，其中 $\$ 1$ \alpha \in [0.1, 10]$ 作为超参数搜索。
$\$ 123,456$ ，⼀次矩阵乘法即可得到，⽆需 SVD。
# 第七步：可选增强⸺加⼊梯度统计量
# 7.1 动机
上⾯的⽅法只⽤了 $\$ 0$ 的⼏何信息。可以叠加梯度的统计信息（类似 Adam 的⼆阶矩）来进⼀步⾃适应：
# 7.2 SVD 基下的 Adam 式⾃适应
在 SVD 基下，对重加权后的分量做 EMA：
```perl
$$ \tilde{t}ilde{G}_{-}t = U^{\wedge}\top G_{-}t, V $$ 
```
```latex
$$ \tilde{tilde{\backslash{Delta}}}\^{\backslash\text{text{raw}}}\{\{ij\}=-|frac{|tilde{G}\{\{t,ij\}\}\{f(\backslash sigma_i)\}$ 
```
```txt
$$ v_{}\{t,ij\} = \beta e t a_2, v_{}\{t-1,ij\} + (1 - \beta e t a_2), (\tilde{\text{tilde}}\{G\}_{}\{t,ij\})^2 $ 
```
```txt
$$ \tilde{tilde}\{\Delta tla\} \{ij\} = -|frac|\{tilde{G}\} \{ij\} \{f(\backslash sigma_i)\backslash cdot\sqrt{sqrt}\{v_{i j} + \backslash delta\} $ 
```
```javascript
$$ \Delta Delta W = \eta;U;\tilde{tilde}\{\Delta Delta\};V^{\wedge}\top$$ 
```
代价：需要维护 $v_{ij} $\$ 5$ 缓冲区（和权重同形），且需要 $U, $\vee \$ 8$ （但可以间隔 $\$ 123$ 步更新）。
# 7.3 是否值得
在 RLVR 中， $\$ 0,0$ ⼏乎不变（论⽂ Figure 4：top- $\boldsymbol { \cdot } \boldsymbol { \mathsf { k } }$ 主⻆旋转 $< 2 ^ { \circ }$ ），所以 SVD 基是稳定的。但 $v_{ij} $\$ 9$ 需要在这个稳定基下积累统计量，这在概念上是合理的。
建议：先不加，⽤纯⼏何⽅法做 baseline。如果效果不够再叠加。
# 第⼋步：是否需要右侧预条件
# 8.1 问题
上⾯的 $\$ 123,456$ $\boldsymbol { \mathfrak { G } } \boldsymbol { \Phi }$ 只做了”左预条件”⸺⽤ $\$ 0$ 的左奇异向量⽅向的结构来加权。
完整的两侧预条件是：
```txt
$$ \Delta Delta W = -(WW^{\wedge}\top + \epsilon \text{psilon} \text{ilon\_L} I)^{\wedge}\{-1\}; G; (W^{\wedge}\top W + \epsilon \text{psilon} \text{ilon\_R} I)^{\wedge}\{-1\} $$$ 
```
右侧 $\$ 123,456,78$ ⽤的是 $\$ 0$ 的右奇异向量⽅向的结构。
# 8.2 分析
左奇异向量对应输出空间（隐藏状态维度）
# 右奇异向量对应输⼊空间
在 Transformer 的线性层 $\$ 1$ 中， $\$ 0$ 的左奇异向量对应输⼊特征的混合⽅向，右奇异向量对应输出特征的混合⽅向。两侧的谱结构都可能有各向异性。
但两侧同时预条件会让更新缩⼩太多（两次除以 $\sigma^2$），可能需要相应增⼤学习率。 建议先只做左侧，验证有效后再考虑两侧。
# 第九步：完整算法伪代码

Algorithm: Geometry-Aware Steepest Descent for RLVR (GASD)


Input:

$\mathsf{W}\in \mathsf{R}^{\wedge}\{\mathsf{n}\times \mathsf{m}\}$ -- 权重矩阵  
n -- 学习率  
ε -- 几何正则化参数（控制off-principal放大程度）  
λ -- weight decay  
K_cg -- CG迭代次数（默认10）

For each step t:

1. 计算梯度
G_t = $\nabla_{-}\mathbb{W}$ L(W_t-1)} 
2. ⽤ CG 解线性⽅程组

求 $\Delta$ 使得 $( \mathbb { W } \textrm {  { W } } ^ { \setminus } \mathbb { T } + \textrm {  { \sf \varepsilon } } \mathbb { I }$ ) $\Delta \ = \ \mathsf { G \_ t }$

-- CG 的每次迭代：

r = G_t - $\bar { \mathsf { W } } \bar { \mathsf { W } } ^ { \wedge } \bar { \mathsf { T } } + \mathsf { \Omega } \bar { \mathsf { g } } \bar { \mathsf { I } }$ ) Δ （残差）

矩阵-向量乘： $\texttt { V }  \texttt { W } ^ { \wedge } \texttt { T } \texttt { p }$ ，然后 $\bar { \mathbb { W } } \texttt { V } + \texttt { \varepsilon } \texttt { p }$ 
标准 CG 更新 α, β, Δ, r, p
重复 K_cg 次
3. 更新权重
W_t = W_{t-1} - $\eta$ ( $\Delta + \lambda$ W_{t-1}) 
# 9.1 ⼯程细节
CG 的初始化：可以⽤上⼀步的 $\Delta$ 作为 warm start，因为连续步之间 $\$ 0$ 变化很⼩
$\epsilon$ 的⾃适应：可以⽤ $\epsilon $=$ \alpha \cdot \text{Tr}(WW^\top)/n$ ⾃动适配不同层的尺度
混合精度： $\$ 0$ 和 $\$ 03$ ⽤ bf16，CG 内部⽤ fp32
与 Adam/1D 参数的混合：对 embedding、LayerNorm 等⾮矩阵参数⽤ Adam，对线性层的$\$ 0$ ⽤ GASD

第⼗步：与已有⽅法的理论关系

<table><tr><td>方法</td><td>更新公式</td><td>度量来源</td><td>对 RLVR 的适配性</td></tr><tr><td>SGD</td><td>$-G$</td><td>欧氏距离</td><td>无结构感知</td></tr><tr><td>Adam</td><td>$-G / \sqrt{v} $</td><td>梯度统计量</td><td>有自适应,无几何</td></tr><tr><td>Muon</td><td>$-U_G V_G^{\wedge}\top$(msign)</td><td>Schatten- \$\infty\supset范数</td><td>各向同性,违反 off-principal</td></tr><tr><td>Shampoo</td><td>$-L^{\wedge}(-1/2)\ G R^{\wedge}(-1/2)$</td><td>梯度二阶矩</td><td>有结构,但基于梯度而非权重</td></tr><tr><td>GASD(本方案)</td><td>$-(WW^{\wedge}\top + \epsilonpsilon epsilonI)^{\wedge}(-1)\ G$</td><td>权重的谱结构</td><td>直接编码 off-principal 偏好</td></tr></table>

关键区别：GASD 是唯⼀⼀个把预条件器建⽴在 $\$ 103$ 本身（⽽⾮梯度或 Fisher）上的⽅法。 这恰好对应 Three-Gate Theory 的核⼼发现⸺RLVR 的优化偏好来⾃预训练权重的⼏何结构，与数据集和 RL 算法⽆关。
# 第⼗⼀步：实验验证计划
# 11.1 ⼩规模验证（第⼀阶段）
模型：DeepSeek-R1-Distill-Qwen-1.5B
任务：DAPO on MATH
对⽐：AdamW, Muon, GASD
观测指标：
训练 reward 曲线
谱漂移 NSS（验证 GASD 是否更好地保持谱结构）
Principal angle 变化
更新 mask 与 principal mask 的重叠度（应为 sub-random）
# 11.2 超参搜索
$\epsilon$：$[0.01, 0.1, 1.0, 10.0] \times \text{mean}(\sigma_i^2)$ 
$K_{cg}$：$[3, 5, 10, 20]$ 
学习率：需要重新搜索（预条件改变了有效步⻓）
# 11.3 消融实验
只做左预条件 vs 两侧预条件
CG 迭代次数对精度和速度的影响
$\epsilon$ 固定 vs 按层⾃适应 vs 训练中退⽕
和 Adam 的⼆阶矩叠加 vs 纯⼏何
# 总结：完整逻辑链
Three-Gate Theory: RLVR 更新受 W 的谱结构引导  
Hessian 不行：RLVR 的 loss 不稳定，Newton 步无意义  
正确度量：用 W 的奇异值结构定义"步长代价"  
加权范数最速下降：price(方向 i) = f(\sigma_i)  
闭式解： $\Delta W = -U \cdot \text{diag}(1/f(\sigma_i)) \cdot U^T \cdot G$ 选 $f(\sigma) = \sigma^2 + \varepsilon$ ，利用 $WW^T = U\Sigma^2 U^T$ 消除 SVD: $\Delta W = -(WW^T + \varepsilon I)^{-1} G$ GPU 友好：用 CG 解线性方程组，每次迭代只需两次矩阵乘法  
工程实现：CG warm start + 按层自适应 $\varepsilon +$ 混合精度