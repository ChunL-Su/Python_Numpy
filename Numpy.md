# Numpy
## Numpy的一些常用的属性
```
import numpy as np

array = np.array([[1,2,3],
                  [2,3,4])
print(array.ndim) # 矩阵的维度
print(array.shape) # 返回(行，列)这样的一根元组
print(array.size) # 返回矩阵中的元素个数，行x列的值
```

## 使用numpy创建一个array
- 1、创建矩阵时可以定义数据类型
```
import numpy as np

array = np.array([1,2,3])
# 也可以定义矩阵数据类型
array1 = np.array([1,2,3], dtype=np.int)
# 常用类型有np.int, np.float16,32,64等
```
- 2、numpy还提供了快速创建全零矩阵，全一矩阵和有序矩阵的方法
```
import numpy as np

array = np.zeros((row，col)) # 全零
array = np.ones((row，col)) # 全一
array = np.arange(3, 9, 2) # [3 5 7]
                           # 起始3，结尾8，步长2
# 还可以通过reshape方法改变其维度
array.reshape((3, 1)) # 返回[[3],
                            [5],
                            [7]]
```
- 3、numpy还提供了一个自动“分割”的方法
```python
import numpy as np

array = np.linspace(1, 10, 5)
# 初始1，结尾10，共5个数字
# 返回[1 3.25 5.5 7.75 10]
```

## 矩阵的基本运算
- 1、相同大小的两个矩阵(各个相同位置的)元素可以进行+ - x / //的运算
- 2、矩阵自己可进行乘方**运算
- 3、矩阵也可直接与数字进行+-x/ << >>等运算
- 4、除了以上进行的基本运算之外还可以进行求sin cos等运算
```python
import numpy as np

array = np.array([1,2,3,4])
b = np.sin(array) # 对array中每个元素进行求sin值
```
- 5、数据的筛选比对
```python
import numpy as np

array = np.arange(4)
print(array < 3)
# 返回[True True True False]
# array 中的元素逐个与3进行比对
```
- 6、以上为普通运算，如果需要使用“矩阵乘法”则：
```python
import numpy as np

a = np.array([[1,1],[2,2]])
b = np.array([[0,1],[2,3]])

c = a * b # 普通乘法
c_dot = np.dot(a, b) # 矩阵乘法
c_dot2 = a.dot(b) # 效果相同
```
- 7、生产指定大小的随机矩阵
```python
import numpy as np

array = np.random.random((2, 4))
# 生产一个两行四列的矩阵，元素大小0~1内随机

np.sum(array) # 返回所有元素的和
np.min(array) # 返回所有元素中最小的元素
np.max(array) # 返回所有元素中最大的元素
np.mean(array) # 返回A的平均值
array.mean() # A的平均值

# 以上四种方法还可以单独对行维度或者列维度进行依次求和，求最小最大值
# 例如
np.sum(array, axis=1) # 返回一个列表，每个元素为array中各个行的和，axis=0则为列和
```
- 8、矩阵中最大值最小值的索引，以及平均值，中位数
```python
import numpy as np

A = np.arange(2, 14).reshape((3,4))
np.argmin(A) # 返回矩阵A中最小值所在的索引
np.argmax(A) # 返回矩阵A中最大值所在的索引

np.median(A) # 返回矩阵A的中位数

np.cumsum(A) # 返回A中元素逐个累加的一个列表

np.diff(A) # 返回各个元素之间的差

np.transpose(A) # 矩阵的转置
print(A.T) # 矩阵的转置

np.clip(A, 5, 9) # 把A中小于5的数字全改为5
                 # 大于9的数字全改为9
```
- 9、转换为一维矩阵
```python
import numpy as np

a = np.arange(2, 14).reshape((3,4))

print(a.flatten()) # 把a矩阵重新拉成一维的一个列表
print(a.flat) # a.flat 返回一个可迭代的对象
```