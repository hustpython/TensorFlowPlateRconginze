# 一些关于CNN的常识性知识 

## 1. 卷积岑conv layer
### 1.1 计算经卷积核之后的输出大小
输入数据维度 W×W

Filter大小 F×F 

步长S

padding像素数目P
(通常padding为0像素)

输出 N

$$ N = (W - F+2P)/S+1$$

>> 例子:
Input volume : 32×32×3
10 5×5 filters with stride 1 , pad 2 
Output volume size:
(32+2×2-5)/1+1 = 32 spatically,so 32×32×10 

### 1.2 计算卷积层中的参数
## 2. 激活函数 
### 2.1 ReLu
![](./doc/relu.png)
## 3. pool 层

### 3.1 max_pool
![](./doc/1.png)

## 4. Full-connect layer
### 4.1 
![](./doc/fullconnect.png)