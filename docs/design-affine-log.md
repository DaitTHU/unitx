# 仿射读数与对数量：设计草案

状态：主体已实施，保留本文作为设计记录；当前公开契约以 affine.md 为准。
依据：2026-10-21 当前源码，以及本文末尾列出的官方资料。

## 1. 目标与基本判断

- Unit 与 Quantity 始终属于线性单位和差值域；Unit(str) 固定返回 Unit。
- AffineUnit 与 AffineQuantity 独立表示坐标及其绝对读数；不继承线性类。
- 数值乘 Unit/AffineUnit 双向构造线性 Quantity；数值 @ 描述符构造 AffineQuantity，含 NumPy 数组。
- AffineQuantity 在转换成 K/W/V 后仍是点；to_quantity(unit) 显式退出到普通物理量。
- degC/degF/dBm/dBW/dBV 的同一拼写可带注册坐标元数据；Unit 只读取其线性差值定义。
- 对数物理映射是非线性的；Affine 表示点与差值结构及相容坐标间的仿射变换。
- 不把非线性转换塞入 Factor 或 ScaledUnit，不建立通用转换图或策略插件。

## 2. 数学模型：一个差值单位，一个原点

线性物理坐标：

    q = q0 + x * delta

对数物理坐标：令 `s` 为差值按 Np 计的数值：

    amplitude: q = q0 * exp(s)
    power:     q = q0 * exp(2*s)

这里 q0 是零读数对应的物理参考量，必须与 q 同量纲；对数参考量还须为正且有限。
线性参考量可以为零、正或负；两个模型不应强行共用参考量的正性校验。

| 单位 | 差值单位 | 零读数对应的物理量 | 映射 |
| --- | --- | --- | --- |
| degC | K | 273.15 K | T = (x + 273.15) K |
| degF | degR | (45967/180) K | T = (5*x/9 + 45967/180) K |
| dBm | dB | 1 mW | P = 10**(x/10) mW |
| dBW | dB | 1 W | P = 10**(x/10) W |
| dBV | dB | 1 V | V = 10**(x/20) V |

单位本身不能推断功率或幅度约定。即使参考量量纲相同，也不能忽略约定差异。
例如相同量纲的两个自定义坐标可以分别采用 10 log10 和 20 log10。

`AffineUnit.delta_unit` 表达对应关系；不把 `Delta dBm = dB` 实现成单位对象的减法。
同一物理量纲、同一映射约定内，转换到另一带原点坐标时，差值缩放由两个 delta_unit 决定，原点差由参考量决定。
同一物理量纲上若使用不同的 power/amplitude 约定，坐标仍可显式转换，
但变换斜率还包含约定的比例，不能只看两个 delta_unit 的换算比。

## 3. Dimension 与用户提出的 LogDimension

`LogDimension(underlying_dimension)` 的有效信息是“这个读数对应什么物理量”。
这与 Astropy LogUnit 保存 physical_unit 的目的相近，但不等于应继承七维 Dimension。

当前 Dimension 具有以下具体约束：

- `__new__` 归一化七个有理指数，并把全零指数返回为 DIMENSIONLESS；仅增加 __init__ 不足以构造子类。
- 实例不可变，普通属性赋值被 _Immutable 阻止。
- 相等和哈希只看七个指数；额外字段不会自动参与身份。
- 乘除、幂、repr、pickle 均只认识七个指数；会丢失新字段或走错误的重建签名。

更根本的代数问题：

- `log(P/P0)` 是无量纲值，underlying_dimension 是物理来源的元数据，不是对量纲本身取数学对数。
- 两个相容 dBm 读数相减得到 dB，原物理量纲消去。
- `dB/m` 是对数差值每长度，不是“某个逆长度的对数”。
- `dB**2` 是对数差值的平方，可用于方差；不等于“对某个物理量平方取对数”。
- 标量乘以一个对数量不改变它的单位；对数量的平方则改变单位。这两种操作必须区分。

推荐：先保持 Dimension 为七维物理量纲；AffineUnit 保存物理参考单位/参考量，
由此得到 physical dimension。dBm 的物理量纲为功率，dBV 为电压；二者 delta_unit 的物理量纲均为一。
普通 Unit 的物理量纲、对数差值的语义，以及 AffineUnit 对应的物理量纲分别处理。

如坚持采用 LogDimension，应把它作为新的语义描述对象，重新定义其相等、组合和转换，
而不是直接套用 Dimension 的继承行为。要闭合支持 dB/m 和 dB**2，还需要混合、幂结构；
这已经是量纲模型重设计，应评估其收益是否超过本版需求。

## 4. 对数差值：log=True 能做什么

建议定义字段使用可读的 `logarithmic: bool = False`，而不是含义模糊的转换回调。
它表示该原子单位属于对数差值；不包含参考量，也不默认功率约定。

相干对数差值以 Np 计，倍率关系为：

    Np: 1
    B:  ln(10)/2
    dB: B/10

这些是对数量之间的乘法换算；恢复原始比值时才执行指数。

仅有 bool 不足以处理复合单位。建议内部从形式单位积计算有理数 log_degree：

    log_degree = sum(exponent for logarithmic atoms)

| 表达式 | 物理量纲 | log_degree |
| --- | --- | --- |
| dB | 1 | 1 |
| dB/m | L^-1 | 1 |
| dB**2 | 1 | 2 |
| dB/Np | 1 | 0 |

这不是新增物理基本量纲，也不扩展为通用 quantity-kind 框架；它是防止无意消除对数语义的有限内部信息。
相加、直接乘法换算及比较要求物理量纲和 log_degree 都相容。
log_degree=0 时允许成为普通物理量；例如 dB/Np 是一个带倍率的普通无量纲单位。

需要明确认可的行为变化：dB 虽然物理上无量纲，但不因此自动接受裸数加法或正弦运算。
`0 dB` 也不与裸数 0 自动相等。这里采取与普通无量纲角度不同的语义边界。
该选择比保留任意无量纲混用更严格，必须写入公开文档。

对数 Unit 的 algebra 仍为形式代数。dB**2、dB*m 都可构造，
但只有物理量纲为一且 log_degree=1 的结果能被直接解释成单个比值。

## 5. 转换 API

推荐保持 `.to()` 在对数差值域内为乘法换算：

    (20*dB).to(B)         -> 2 B
    (20*dB).to(Np)        -> approximately 2.302585 Np
    (20*dB).to('')        -> error: explicit ratio conversion required

上式最后一条取代前轮讨论中“返回相干无量纲系数”的提案，避免隐式剥除对数语义。
需要系数时用 `.value_in(Np)`；恢复物理比值则明确选约定：

    (20*dB).to_ratio(kind='power')        -> approximately Quantity(100, '')
    (20*dB).to_ratio(kind='amplitude')    -> approximately Quantity(10, '')
    Quantity(100, '').to_log(dB, kind='power')

kind 必填，没有“电压自动猜 20、功率自动猜 10”的规则。
先完成 dB/km * km 得到对数差值，再恢复比值；不对 dB/km 直接执行指数。

AffineQuantity 的坐标已包含物理映射，点的 `.to()` 保留点类别：

    (0 @ dBm).to(mW)       -> 1 mW
    (30 @ dBm).to(dBW)     -> 0 dBW
    (1 @ mW).to(dBm)       -> 0 dBm
    (0 @ dBV).to(V)        -> 1 V
    (20 @ degC).to(K)      -> 293.15 K

先匹配单位类别，再选择转换路径；不把所有转换都改成经过浮点 SI 中间值。
同一坐标保持 magnitude/数组对象，整数十进制参考比应有精确路径。
dBm 到 dBW 直接加减 30，避免先指数后对数。
对任意自定义参考比，若必须求对数则明确进入近似数值域，不承诺精确往返。

## 6. 点、差值和普通物理量的运算

| 操作 | 结果 |
| --- | --- |
| AffineQuantity - AffineQuantity | 左侧 delta_unit 的 Quantity 差值 |
| AffineQuantity + Quantity；Quantity + AffineQuantity | AffineQuantity 点 |
| AffineQuantity - Quantity | AffineQuantity 点 |
| AffineQuantity + AffineQuantity；Quantity - AffineQuantity | 报错 |
| 点的乘除、矩阵乘法、幂、负号、绝对值 | 报错；先 to_quantity(unit) |

K 的含义由对象类别决定，不由上下文决定：Quantity(300, 'K') 是差值，
AffineQuantity(300, 'K') 或 300 @ si.K 是绝对温度点。to() 保留点类别。
例如 (300 @ si.K) - (20 @ si.degC) 为 6.85 K 温差；
Quantity(300, 'K') - Quantity(20, 'degC') 则为 280 K。

数值 @ Unit 或数值 @ AffineUnit 是点构造；Unit @ 数值不是构造。
两个 Quantity 之间的 @ 保持已有矩阵乘法。裸 ndarray @ Unit 在 NumPy 分派入口构造点，
不将单位传入矩阵乘法内核。普通乘法构造差值并满足交换顺序。
负读数可以直接构造；对已经存在的点取负仍无定义。

## 7. 单位对象、解析与注册

Unit、AffineUnit 独立继承 _Immutable；ScaledUnit 继承 Unit。
Quantity.unit 为 Unit，AffineQuantity.unit 为 AffineUnit。

- Unit('degC') 是摄氏差值单位，Unit('K') 是开尔文差值单位；二者换算无偏移。
- UnitParser.parse() 仍返回 Monomial[SingleUnit]，允许 degC/m、degC/degC、degC**0。
- AffineQuantity(value, 'degC') 或 value @ Unit('degC') 查找绝对摄氏坐标。
- 没有额外坐标元数据的普通物理单位，作为点坐标时具有零原点。
- AffineUnit 构造器不接受字符串，不提供恒抛错的 factor 属性。
- si.degC 等保存 AffineUnit 描述符；乘除和幂投影到线性单位，@ 取完整坐标。
- Quantity 构造和 to/value_in 接受描述符但只取线性部分，存储的 unit 仍为 Unit。
- 已注册描述符投影保留拼写；匿名描述符使用 delta_unit，不能在构造器与运算符之间分叉。

共享 UnitDefinition 始终以 factor/dimension/log_degree 描述线性差值单位。
可选 reference 保存坐标原点的普通物理量数据，delta/kind 描述刻度和对数约定。
注册核心不导入 Quantity/AffineQuantity。注册校验和失败原子性沿用同一套流程。

自定义入口为 AffineUnit(reference, /, *, at=0, delta_unit=None, kind=None)，
表示新坐标读数 at 对应 reference。reference 可以是普通物理 Quantity 或另一 AffineQuantity。
例如 AffineUnit(0 @ si.degC, at=32, delta_unit=si.degR) 定义华氏温标。
线性差值单位默认取 Quantity 的 unit 或 AffineQuantity 的 delta_unit；
对数坐标要求显式指定 delta_unit 和 kind。构造立即规范化 origin，不保存引用链。

define(symbol, name, ...) 注册线性差值拼写及坐标元数据，返回 AffineUnit。
普通 Quantity.define() 定义线性大小；AffineQuantity 不提供该入口。
带显式 ScaledUnit 的参考和差值单位须先注册，不能在存储表达式时丢失倍率。
内置偏移坐标的拼写保持 prefixable=False；这是当前目录策略。
B/Np 的前缀和 dB 的 deci+B 身份保持原有规则。

## 8. 比较与哈希

- 同一 affine 单位先按 magnitude 原生规则比较，保留 NaN、Decimal 信号和数组掩码。
- 相容线性 AffineQuantity 坐标（包括 K/degR）之间比较，使用精确缩放和有符号偏移；禁止只凭 factor 相同直接比较读数。
- 同一 log convention 的 affine 坐标之间，优先直接仿射转换并比较。
- affine 对数读数与 W/V 的跨非线性相等首版要求显式转换；默认相等为 False，排序报错。
  这是支持边界，不表示物理量不同；避免浮点指数结果决定所谓精确数学身份。
- 不用 isclose 实现 ==。跨参考比若涉及近似坐标转换，不能宣称严格的跨单位数学相等；首版精确比较需限制在可精确对齐的坐标族。
- AffineQuantity 全部不可哈希，包括标量，避免未经验证的规范化破坏相等/哈希契约。
- 普通及对数差值 Quantity 的哈希继续精确；加入 log_degree 语义，确保相等者同哈希。
- 数组 Quantity 继续不可哈希。AffineUnit 对象本身不可变、可哈希，身份不能只看 delta 和物理量纲。

跨非线性比较的支持边界比 Pint/Astropy 的部分自动行为更保守；必须作为公开限制审阅。
若要首版全面支持 dBm == W，需要额外的精确指数表达及相等规范化设计，不能顺手转 float。

## 9. isclose 与容差

绝对容差属于 Quantity 差值域；K、degC、degF、dB 等线性单位按量纲与对数度数校验。
不能使用 AffineQuantity 点作为容差。非零裸容差不能获得目标坐标单位。
AffineQuantity.isclose 默认 rel_tol=0、abs_tol=0；非零相对容差明确报错。
需要物理相对误差时，先 to_quantity('K'/'W'/'V')，再用普通 Quantity.isclose。
NumPy 保持自己的 rtol/atol 默认值；比较点时须使用 rtol=0 和合适的差值 atol。
两种 Quantity 的类别不隐式互换，不能用 isclose 掩盖身份差异。

## 10. 数值域、精确性和端点

- 摄氏和华氏内置常量使用 Fraction，偏移带符号，不用正 Factor 存储 offset。
- 仿射变换先组合精确系数再作用于标量；Decimal 精确构造结果后按当前上下文舍入一次。
- 不经两次转换制造 Decimal 双重舍入；同一坐标转换保留 magnitude。
- 对数和指数按支持的数值域求值：Python int/Fraction 通常进入 float；Decimal 使用自身 ln/exp 和上下文。
- Fraction 输入不保证非线性输出仍为 Fraction。仅对可证明的简单精确路径保留精确值，不能用容差猜整指数。
- 参考量必须是有限实标量；对数参考严格为正。首版不接受数组参考或复杂参考。
- 读数支持实数与普通实数数组；对数/仿射新路径不接受复数，不改变普通复数 Quantity。
- 线性物理量转对数：负输入报错，零映射为负无穷，正无穷为正无穷，NaN 传播。
- 对数读数转物理量：负无穷为零，正无穷为正无穷；遵循标量/Decimal/NumPy 各自溢出和错误规则。
- 不额外强制“温度必须非负”，因为本库的单位变换不应擅自承担全部物理模型约束。

已按用户决定采用 `ln10 = SymbolicConstant('ln(10)', math.log(10))`。
以下为实施前讨论时的数学表示考量，保留作记录：
不能用 Factor(math.log(10)) 把近似量当精确常量；若增加 SymbolicConstant，必须遵循既有数学独立性契约，
不能从浮点近似推出独立性。也不应为了这个常量引入通用符号表达式框架。
实施前需专门审阅这个表示问题；B<->Np 的近似数值结果与结构精确性要分别说明。

## 11. NumPy 支持矩阵

AffineQuantity 不继承普通 Quantity 的数值代数，按下表分派；对数差值除显式限制外沿用乘法单位规则。

| 操作 | AffineQuantity 规则 |
| --- | --- |
| 构造、索引、迭代、reshape、transpose、flatten | 保留单位和原有视图/复制行为 |
| concatenate、stack、选择 | 相容 affine 坐标先转第一个坐标；普通物理量混入要求显式转换 |
| add/subtract、比较 | 与 Python 操作符共享点/差值规则 |
| min/max、argmin/argmax、isfinite/isnan | 保留点、返回索引或裸布尔值 |
| mean | 返回点；在坐标域求均值 |
| median | 返回点；在坐标域求中位数，包括偶数样本的坐标中点 |
| std | 返回 delta_unit |
| var | 返回 delta_unit**2 |
| sum、cumsum、add.reduce/accumulate、prod | 明确拒绝 |
| multiply/divide、matmul、power、sqrt、abs、negative | 明确拒绝 |
| exp/log/sin/cos 等 | 明确拒绝；使用显式物理量/比值转换 |
| modf、frexp、floor/ceil 等 | 首版不继承支持；逐项定义前明确拒绝 |

dBm 数组的 mean 是读数的均值，对应功率的几何平均，不是平均功率；
平均功率应先 to_quantity('W') 求 mean，再显式构造 W 点转 dBm。不能为了用户可能想求功率平均而暗改 mean 的数值域。
对数坐标 median 的偶数样本中点也与先转物理量求 median 不同，应在示例中说明。
std/var 是坐标统计，并不自动表达非线性误差传播后的物理功率标准差。

mean/std/var 的 where、axis、ddof、mean= 等参数继续逐项校验；mean= 若支持，必须是相容的点。
未知函数继续按 NumPy 协议分派；已识别但非法的点运算抛明确错误，禁止解包 value 后兜底。

## 12. 写入、out 和别名

- 目标 affine 单位、dtype、shape 与 ndarray 对象固定。
- 点写入点目标，差值写入差值目标；std 不可写入 degC/dBm 的 out。
- 可显式 `.to()` 物理单位，不代表赋值时自动改变点/差值语义。混合表示写入首版要求显式转换。
- q_affine += delta / -= delta 可就地写回，保持原单位；q_affine -= point 产生差值，不能写回点目标，应报错。
- q_affine *= scalar 等明确拒绝；Python 标量也不得回退成意外的读数倍乘。
- 转换、非线性定义域、shape、casting 和可写性检查在提交写入前完成。
- 继续 temporary staging，验证切片共享、重叠输入输出、只读数组及错误不修改目标。
- where 为假的元素不参与不必要的非线性计算或定义域检查；需要在内核中落实，不能只掩码最终 copy。
- 不提供 out 的 where 结果仍遵循已声明的 NumPy 未初始化位置语义。

## 13. 类型、展示和序列化

- Quantity.unit -> Unit；AffineQuantity.unit -> AffineUnit；两类构造分别保留 magnitude 类型 T。
- 仿射/非线性换算的结果联合类型应如实标注，不承诺 Self 或保留原 dtype。
- 不给 Quantity 添加静态运行时单位参数，不试图让类型系统证明所有量纲关系。
- NumPy 仍为可选；核心类型消费者无 NumPy 时不能出现缺失导入或 Unknown。
- 文本、LaTeX、HTML、siunitx 只展示存储读数及当前坐标，不隐式转物理量或执行 log/exp。
- 明确 dBm 为完整标签；delta 结果显示 dB，温差规范输出 K/degR。
- repr、copy、pickle 保留参考、delta、log convention、身份；不能只保存 factor。
- 动态定义跨进程恢复仍要求接收端已登记；不借本版顺便设计完整交换格式。

## 14. 修改范围与验证

主要修改点：unitregistry.py、singleunit.py 的类型边界、unit.py / 新 affineunit.py、
quantity.py、新 affinequantity.py、_numpy.py、_display.py、命名空间、公开文档及类型用例。
Monomial 的通用形式代数保持原样；Dimension 七维结构按推荐方案保持原样。

关键验收应包括：

1. 精确温标参考点及往返：0/100 degC、32/212 degF、K/degR、Fraction、Decimal 各精度上下文。
2. dBm/dBW 直接偏移、dBV 幅度约定、B/dB/Np 缩放、scaled dimensionless 输入。
3. 点/差值全操作矩阵及双向操作数；明确验证 K 的点与差值类别。
4. 普通 Unit 的摄氏差值可复合、消去和取零次幂；点运算则按类别拒绝。
5. dB/m、dB**2、dB/Np 的 log_degree，比较与哈希一致性。
6. NaN/Inf/零/负数、溢出、非线性定义域和 Decimal 规则。
7. NumPy 两种导入顺序、最低/当前支持版本、零维/空/非连续/只读数组，mean/std/var 的不同结果单位。
8. out、setitem、原位更新、where、共享切片与失败前后缓冲区逐值比较。
9. 相等的对称性/传递性及哈希；尤其不可用近似跨坐标对齐破坏 == 的契约。
10. 无 NumPy/有 NumPy 的安装后类型消费者、repr/pickle 和展示无可选导入副作用。

设计阶段不运行测试，也不以现有测试通过证明新能力已经支持。

## 15. 业界参考与本项目取舍

- Pint 非乘法温标：差值单位、点加差值、点直接相加的歧义、默认拒绝部分乘法。
  本项目采用其区分原则，但按本轮讨论使用 K/degR 作为规范差值，并明确 K 的交互解释。
  https://pint.readthedocs.io/en/stable/user/nonmult.html
- Pint 对数单位：支持物理单位恢复，但 dB 默认功率约定，复合对数支持仍有明确限制。
  本项目不照搬默认约定或复合仿射计算，而让纯 dB 恢复比值时显式选择约定。
  https://pint.readthedocs.io/en/stable/user/log_units.html
- Astropy LogUnit：保存 physical_unit/function_unit，支持 physical 恢复；专门限制函数支持。
  本项目借鉴物理来源与对数读数分层，不照搬其专用 Quantity 类及对数点相加时组合物理单位的规则。
  https://docs.astropy.org/en/stable/units/logarithmic_units.html
  https://docs.astropy.org/en/stable/api/astropy.units.LogUnit.html
- Boost.Units absolute：点与向量区别由包装类型表达，两个绝对值相减得到差值。
  本版不选择其全局点身份模型；显式恢复 K/W/V 后按普通物理量处理。
  https://www.boost.org/doc/libs/latest/doc/html/boost_units/Reference.html
- UCUM special units：非比例坐标不属于普通乘法单位群，使用双向函数恢复物理量。
  本项目采用职责边界，不引入任意用户转换函数或通用转换图。
  https://ucum.org/ucum
- NIST：Np、B、dB 的幅度/功率定义，以及 1 B = ln(10)/2 Np。
  https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-8#8.7

## 16. 当前已确定的选择

1. Dimension 保持七维物理量纲；log_degree 单独跟踪对数差值语义。
2. Unit 与 Quantity 始终是线性的；AffineUnit 与 AffineQuantity 独立表示坐标和点。
3. Unit(str) 固定返回 Unit；value * unit 是差值，value @ unit 是点。
4. 点的 to() 保留类别；to_quantity(unit) 显式恢复可参与普通物理运算的量。
5. reference/at 描述对应点，origin 存储规范化零点，不建立引用链。
6. 点无哈希；对数跨非线性精确比较及非零相对容差保留明确边界。
7. ln10 使用 SymbolicConstant；数值求值与符号精确性分开。

具体支持边界、示例和迁移方式见 affine.md。
