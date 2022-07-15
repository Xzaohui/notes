# python
## deque 双向队列 

appendleft,extendleft

## 堆 heapq
```python
def __lt__(self, other):
            return self.val < other.val
        ListNode.__lt__ = __lt__
```
让类也能排序做对比建堆 https://leetcode.cn/problems/merge-k-sorted-lists/

```py
import heapq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        #要统计元素出现频率
        map_ = {} #nums[i]:对应出现的次数
        for i in range(len(nums)):
            map_[nums[i]] = map_.get(nums[i], 0) + 1
        
        #对频率排序
        #定义一个小顶堆，大小为k
        pri_que = [] #小顶堆
        
        #用固定大小为k的小顶堆，扫面所有频率的数值
        for key, freq in map_.items():
            heapq.heappush(pri_que, (freq, key))
            if len(pri_que) > k: #如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
                heapq.heappop(pri_que)
        
        #找出前K个高频元素，因为小顶堆先弹出的是最小的，所以倒序来输出到数组
        result = [0] * k
        for i in range(k-1, -1, -1):
            result[i] = heapq.heappop(pri_que)[1]
        return result
```
前K个高频词 https://leetcode.cn/problems/top-k-frequent-elements/

## strip
Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。

str.strip([chars]);

还有lstrip，rstrip

注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
## match、search、findall、finditer
都可以用r''正则表达式来匹配，也可以是普通的字符串

re.findall(r'',str)

^：匹配字符串开头，$匹配字符串结尾

match方法从头开始找，找到就返回，否则为None，只匹配一次（必须开头就有这个字符串），用group()可以获取匹配的字符串

search从头依次搜索，只匹配一次

findall方法：返回列表，匹配所有，*获取所有匹配的字符串，去掉了列表结构

finditer：返回string中所有相匹配的全部字串，返回形式为迭代器。

## 查找方法
字符串序列.find(子串,开始位置下标,结束位置下标)，返回这个子串开始的位置下标，否则-1

字符串序列.index(子串,开始位置下标,结束位置下标)，返回这个子串开始的位置下标

字符串序列.count(子串,开始位置下标,结束位置下标)，返回某个子串在字符串中出现的次数

rfind()：和find()功能相同，但查找方向从右侧开始

rindex()：和index()功能相同，但查找方向从右侧开始
## bisect
查找： bisect.bisect/bisect_left/bisect_right(array, item)

插入： bisect.insort/insort_left/insort_right(array,item)
## nonlocal/global
```python
def dome_fun():
    num = 0
    def dome_fun_1():
        nonlocal num
        num += 1
        return num
    return num

count = 0
def global_test():
    global count
    count += 1
    print(count)
global_test()
```
1. 作用对象不同：

    nonlocal作用于外部内嵌函数的变量；

    global作用于全局变量。

2. global可以改变全局变量，同时可以定义新的全局变量；nonlocal只能改变外层函数变量，不能定义新的外层函数变量，并且nonlocal也不能改变全局变量。

3. 声名：

    global声名此变量为全局变量；nonlocal声名此变量与外层同名变量为相同的变量。

4. 使用的范围不同：

    global关键字可以用在任何地方，包括最上层函数中和嵌套函数中；

    nonlocal关键字只能用于嵌套函数中，并且外层函数中必须定义了相应的局部变量，否则会发生错误
## zip
```python
zip(iterable1, iterable2, ...)
#54. 螺旋矩阵
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if len(matrix)==1:
            return list(matrix[0])
        return list(matrix[0])+list(self.spiralOrder(list(zip(*(matrix[1:])))[::-1]))
```

# dfs
```python
def dfs(self,res,str,l,r,n):
        if l>n or r>n or r>l:
            return
        if l==n and r==n:
            res.append(str)
            return
        self.dfs(res,str+"(",l+1,r,n)
        self.dfs(res,str+")",l,r+1,n)
   def generateParenthesis(self, n: int) -> List[str]:
        res=[]
        self.dfs(res,"",0,0,n)
        return res
```

```py
def letterCombinations(self, digits: str) -> List[str]:
        dic={2:"abc",3:"def",4:"ghi",5:"jkl",6:"mno",7:"pqrs",8:"tuv",9:"wxyz"}
        if len(digits)==0:
            return []
        if len(digits)==1:
            return list(dic[int(digits)])
        else:
            return [i+j for i in self.letterCombinations(digits[0]) for j in self.letterCombinations(digits[1:])]

```
全排列+剪枝

https://leetcode.cn/problems/generate-parentheses/

https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

# bfs
```py
search_list=[]
deep=0
search_list.append((beginWord,0))

while len(search_list):
    word,deep = search_list.pop(0)
    if word == endWord:
        deep+=1
        break
    for i in range(len(word)):
        for j in range(26):
            new_word = word[:i] + chr(ord('a') + j) + word[i+1:]
            if new_word in wordList:
                search_list.append((new_word,deep+1))
                wordList.remove(new_word)
```

双向BFS即是选择短的list去bfs
```py
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0
        l = len(endWord)
        ws = set(wordList)
        head = {beginWord}
        tail = {endWord}
        tmp = list('abcdefghijklmnopqrstuvwxyz')
        res = 1
        while head:
            if len(head) > len(tail):
                head, tail = tail, head
            q = set()
            for cur in head:
                for i in range(l):
                    for j in tmp:
                        word = cur[:i] + j + cur[i+1:]
                        if word in tail:
                            return res + 1
                        if word in ws:
                            q.add(word)
                            ws.remove(word)
            head = q
            res += 1
        return 0
```
https://leetcode.cn/problems/word-ladder/



# 分治
```py
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        n = len(lists)

        def merge(left, right):
            if left > right:
                return
            if left == right:
                return lists[left]
            mid = (left + right) // 2
            l1 = merge(left, mid)
            l2 = merge(mid + 1, right)
            return mergeTwoLists(l1, l2)

        def mergeTwoLists(l1, l2):
            if not l1 or not l2:
                return l1 or l2
            if l1.val < l2.val:
                l1.next = mergeTwoLists(l1.next, l2)
                return l1
            else:
                l2.next = mergeTwoLists(l1, l2.next)
                return l2
```

分治+递归合并 https://leetcode.cn/problems/merge-k-sorted-lists/



# 最长公共前缀
```py
def longestCommonPrefix(self, strs):
        if not strs: return ""
        s1 = min(strs)
        s2 = max(strs)
        for i,x in enumerate(s1):
            if x != s2[i]:
                return s2[:i]
        return s1

def longestCommonPrefix(self, strs):
        if not strs: return ""
        ss = list(map(set, zip(*strs))) # zip压缩，set去重
        res = ""
        for i, x in enumerate(ss):
            x = list(x)
            if len(x) > 1:
                break
            res = res + x[0]
        return res
```

# 无重复字符的最长子串
```py
def lengthOfLongestSubstring(self, s: str) -> int:
    st = {}
    i, ans = 0, 0
    for j in range(len(s)):
        if s[j] in st:  #字符再次出现
            i = max(st[s[j]], i) # 看上一次出现的位置是在当前起点的前面还是后面，前面不用动，后面要向前移动
        ans = max(ans, j - i + 1)
        st[s[j]] = j + 1 #记录每一个字符最后出现的位置
    return ans
```

# 递归
https://lyl0724.github.io/2020/01/25/1/ 


# 基于比较操作的排序算法平均时间复杂度的下界为O(n log n)，最坏情况下为O(n^2)，空间复杂度为O(n)。

# 前中后缀转换 
中缀转后缀转换过程需要用到栈，具体过程如下：

从左到右扫描字符串

1）如果遇到操作数，我们就直接将其输出（输出我们用队列保存）。

2）如果遇到操作符，当栈为空直接进栈，不为空，判断栈顶元素操作符优先级是否比当前操作符小，小的话直接把当前操作符进栈，不小的话栈顶元素出栈输出，直到栈顶元素操作符优先级比当前操作符小

3）遇到左括号时我们也将其放入栈中。

4）如果遇到一个右括号，则将栈元素弹出，将弹出的操作符输出直到遇到左括号为止。注意，左括号只弹出并不输出。

5）如果我们读到了输入的末尾，则将栈中所有元素依次弹出。

中缀转前缀转换过程同样需要用到栈，具体过程如下：

将中缀表达式转换为前缀表达式：

(1) 初始化两个栈：运算符栈S1和储存中间结果的栈S2；

(2) 从右至左扫描中缀表达式；

(3) 遇到操作数时，将其压入S2；

(4) 遇到运算符时，比较其与S1栈顶运算符的优先级：

    (4-1) 如果S1为空，或栈顶运算符为右括号“)”，则直接将此运算符入栈

    (4-2) 否则，若优先级比栈顶运算符的较高或相等，也将运算符压入S1

    (4-3) 否则，将S1栈顶的运算符弹出并压入到S2中，再次转到(4-1)与S1中新的栈顶运算符相比较；

(5) 遇到括号时：

    (5-1) 如果是右括号“)”，则直接压入S1；

    (5-2)如果是左括号“(”，则依次弹出S1栈顶的运算符，并压入S2，直到遇到右括号为止，此时将这一对括号丢弃；

(6) 重复步骤(2)至(5)，直到表达式的最左边；

(7) 将S1中剩余的运算符依次弹出并压入S2；

(8) 依次弹出S2中的元素并输出，结果即为中缀表达式对应的前缀表达式。

前缀运算 左向右扫描，数字入栈，遇到运算符弹出上面两个数字运算，后再次入栈

后缀运算 右向左扫描，其他一样


# 树

## 前中后序

### 递归
访问和递归项的位置

### 迭代

借助栈，先将根节点放入栈中，然后将右孩子加入栈，再加入左孩子。再pop一个后一样将右孩子加入栈，再加入左孩子

中续先左树入栈，pop后右树入栈

**后序遍历，先序遍历是中左右，后续遍历是左右中，那么我们只需要调整一下先序遍历的代码顺序，就变成中右左的遍历顺序，然后在反转result数组，输出的结果顺序就是左右中了**

## 二叉树
二叉树共父节点 https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/comments/
```py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root==p or root==q or root==None:
            return root
        r=self.lowestCommonAncestor(root.right,p,q)
        l=self.lowestCommonAncestor(root.left,p,q)
        if r!=None and l!=None:
            return root
        elif r!=None:
            return r
        elif l!=None:
            return l
        else:
            return None
```

## 二叉搜索树
中序遍历有序

删除某一节点
```py
def deleteNode(root, key):
    if not root: return None;
    if root.val > key:
        root.left = deleteNode(root.left, key)
    elif root.val < key:
        root.right = deleteNode(root.right, key)
    else:
        if not root.left or not root.right:
            root = root.left if root.left else root.right
        else:
            cur = root.right
            while cur.left: cur = cur.left #右子树的最小值，也可用左子树的最大值
            root.val = cur.val
            root.right = deleteNode(root.right, cur.val) 
        
    return root
```

## 平衡二叉树

```py
# left _rotation
def left_rotation(root):
    temp=root.right
    root.right=temp.left
    temp.left=root
    updateHeight(root)
    updateHeight(temp)
    return temp

# right _rotation
def right_rotation(root):
    temp=root.left
    root.left=temp.right
    temp.right=root
    updateHeight(root)
    updateHeight(temp)
    return temp

```

|mode |判定条件|调整方法|
|----- |-------------|--------------|
|LL    |Balance(root)=2,Balance(root.left)=1   |  right_rotation(root)      |
|LR    |  Balance(root)=2,Balance(root.left)=-1   |   left_rotation(root.left),  right_rotation(root)     |
|RR    | Balance(root)=-2,Balance(root.left)=-1  |   left_rotation(root)    |
|LR    |  Balance(root)=-2,Balance(root.left)=1    |  right_rotation(root.right),left_rotation(root)    |

在增删的时候进行检查和调整



## 红黑树

前身是4阶B树
## 公祖问题
    
```py
def lowestCommonAncestor(root, p, q):
    if root is None or root == p or root == q: #如果root是p或者q，那么直接返回，若p或q是公祖则后面也不用找了
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:      # 如果左右子树都不为空，说明p和q分别在左子树和右子树上，那么这个节点就是公共祖先
        return root
    return left if left else right # 如果左右子树有一个为空，说明p和q在同一侧，返回前一侧的子树
```
## Trie 树
也叫“字典树”。顾名思义，它是一个树形结构。它是一种专门处理字符串匹配的数据结构，用来解决在一组字符串集合中快速查找某个字符串的问题。

Trie 树的本质，就是利用字符串之间的公共前缀，将重复的前缀合并在一起

### 构造Trie树
![](https://img-blog.csdnimg.cn/20210129092521661.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NzI1MTk5OQ==,size_16,color_FFFFFF,t_70)

实际上构造Trie树，是利用数组（每个节点是字符值+下一字符的索引）

![](https://img-blog.csdnimg.cn/20210129093250226.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NzI1MTk5OQ==,size_16,color_FFFFFF,t_70)

### 时间复杂度
构建Trie树时间复杂度是 O(n)（n是Trie树中所有元素的个数）

查询Trie树时间复杂度是 O(k)（k 表示要查找的字符串的长度）

### 缺点
每个数组元素要存储一个 8 字节指针（或者是 4 字节，这个大小跟 CPU、操作系统、编译器等有关），且指针对CPU缓存并不友好。

在重复的前缀并不多的情况下，Trie 树不但不能节省内存，还有可能会浪费更多的内存。（比如，此种字符串第三个字符只有两种可能，而它要维护一个长26的数组。这还只是考虑纯字母的情况，如果是复合型字符串，则会浪费更多空间）

### 解决办法：
方法一： 将每个节点中的数组换成其他数据结构，比如有序数组、跳表、散列表、红黑树等。

    假设我们用有序数组，数组中的指针按照所指向的子节点中的字符的大小顺序排列。
    通过二分查找的方法，快速查找到某个字符应该匹配的子节点的指针。（这就不用维
    护一个上述26的数组，只需要维护两个可能的字符数组）当然，这样为了维护数组顺
    序，插入元素效率较慢。

方法二：缩点优化

### 匹配算法
单模式串匹配算法，是在一个模式串和一个主串之间进行匹配，也就是说，在一个主串中查找一个模式串。

多模式串匹配算法，就是在多个模式串和一个主串之间做匹配，也就是说，在一个主串中查找多个模式串。（AC自动机）

AC自动机是KMP和trie的结合体。KMP算法适用于单模式串的匹配，而AC自动机适合多模式串的匹配，KMP是AC自动机的特殊情况。

在KMP中，初始值nex[0] = nex[1] = 0, 另外如果我们需要求出nex[i]，则需要用到nex[0]-nex[i-1],在AC自动机中，如果我们需要求出第i层点的nex值，则需要使用到i-1层（包括）前面的nex值。

由于每次向下求一层的nex，所以用bfs。

### 应用

Trie 树只是不适合精确匹配查找，这种问题更适合用散列表或者红黑树来解决。 Trie 树比较适合的是查找前缀匹配的字符串

Trie 树的这个应用可以扩展到更加广泛的一个应用上，就是自动输入补全，比如输入法自动补全功能、IDE 代码编辑器自动补全功能、浏览器网址输入的自动补全功能等等。
# 贪心

## 分发糖果

评分高的孩子获得更多的糖果。

一次是从左到右遍历，只比较右边孩子评分比左边大的情况。

一次是从右到左遍历，只比较左边孩子评分比右边大的情况。

```py
class Solution:
    def candy(self, ratings: List[int]) -> int:
        candyVec = [1] * len(ratings)
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candyVec[i] = candyVec[i - 1] + 1
        for j in range(len(ratings) - 2, -1, -1):
            if ratings[j] > ratings[j + 1]:
                candyVec[j] = max(candyVec[j], candyVec[j + 1] + 1)
        return sum(candyVec)
```

# dp

## 编辑距离
```py
def minDistance(self, word1: str, word2: str) -> int:
        word1=list(word1)
        word2=list(word2)
        n,m=len(word1),len(word2)
        dp=[[0]*(len(word2) + 1) for _ in range(len(word1) + 1)]
        for i in range(1,n+1):
            dp[i][0]=i
        for i in range(1,m+1):
            dp[0][i]=i
        for i in range(1,n+1):
            for j in range(1,m+1):
                if word1[i-1]==word2[j-1]:
                    dp[i][j]=dp[i-1][j-1] #相同直接不用增加操作数
                else:
                    dp[i][j]=min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1#左、上、左上的情况需要多变换一次
        return dp[-1][-1]
```

## 跳表问题

```py
for i in range(len(nums)-2,-1,-1):
    jump[i]=min([jump[j] for j in range(i+1,min(i+nums[i]+1,len(nums)))])+1
return jump[0]
```
## 买卖股票
### 只买卖一次
```py
def maxProfit(self, prices: List[int]) -> int:
    length = len(prices)
    if len == 0:
        return 0
    dp = [[0] * 2 for _ in range(length)]
    dp[0][0] = -prices[0]
    dp[0][1] = 0
    for i in range(1, length):
        dp[i][0] = max(dp[i-1][0], -prices[i]) # 只买卖一次，不会把新的盈利再买卖。
        dp[i][1] = max(dp[i-1][1], prices[i] + dp[i-1][0])
    return dp[-1][1]
```


### 多次买卖（含手续费）
```py
def maxProfit(self, prices: List[int]) -> int:
    length = len(prices)
    dp = [[0] * 2 for _ in range(length)]
    dp[0][0] = -prices[0] #第i天手上有股票时的最大收益
    dp[0][1] = 0 #第i天手上无股票时的最大收益
    for i in range(1, length):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i]) #注意这里是和121. 买卖股票的最佳时机唯一不同的地方
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i]) #- fee 就是有手续费
    return dp[-1][1]
```
### 规定次数买卖
```py
def maxProfit(self, k: int, prices: List[int]) -> int:
    if len(prices) == 0:
        return 0
    dp = [[0] * (2*k+1) for _ in range(len(prices))]
    for j in range(1, 2*k, 2):
        dp[0][j] = -prices[0]
    for i in range(1, len(prices)):
        for j in range(0, 2*k-1, 2):
            dp[i][j+1] = max(dp[i-1][j+1], dp[i-1][j] - prices[i])
            dp[i][j+2] = max(dp[i-1][j+2], dp[i-1][j+1] + prices[i])
    return dp[-1][2*k]
```
## 背包

### 01背包问题

01背包：必须倒序遍历数组。

完全背包：顺序遍历。

分割 等和子集、最相似子集：背包大小为和的一半。

目标和，有+-符号组合：转换为letf-right=traget，left+right=sum，left=sum+target/2。再转化为01背包。

一和零：二维01背包问题，两个容量限制。注意两个都要倒序计算。
```py
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp=[[0]*(n+1) for _ in range(m+1)]
        for i in range(len(strs)):
            for j in range(m,-1,-1):
                for k in range(n,-1,-1):
                    if j >= strs[i].count("0") and k >= strs[i].count("1"):
                        dp[j][k]=max(dp[j][k],dp[j-strs[i].count("0")][k-strs[i].count("1")]+1)

        return dp[-1][-1]
```
### 完全背包问题

如果求组合数就是外层for循环遍历物品，内层for遍历背包。

如果求排列数就是外层for遍历背包，内层for循环遍历物品。

518.零钱兑换问题，没有顺序要求。377.组合总和有顺序要求。
```py
for coin in coins: #零钱兑换问题
    for i in range(1,amount+1):
        if coin <=i:
            dp[i]+=dp[i-coin]

for i in range(1,target+1):#组合总和
            for num in nums:
                if num <=i:
                    dp[i]+=dp[i-num]

        return dp[-1]
```
### 多重背包问题
有N种物品和一个容量为V 的背包。第i种物品最多有Mi件可用，每件耗费的空间是Ci ，价值是Wi 。求解将哪些物品装入背包可使这些物品的耗费的空间 总和不超过背包容量，且价值总和最大。

转化成01背包，把每个物品扩展Mi次。

## 成环就考虑两种，一不取头，二不取尾

## 树形后序遍历 dp
337. 打家劫舍 III

https://leetcode.cn/problems/house-robber-iii/
```py
def trob(root):
   if root == None:
       return [0,0]
   r=trob(root.right)
   l=trob(root.left)
   return [root.val+r[1]+l[1],max(r)+max(l)] #不抢也要返回他的max
return max(trob(root))
```

# 最短路径问题

## dijkstra 算法
```py
dijkstra(graph,d[], start):
    for each vertex v in graph:
        d[v] = inf
    d[start] = 0
    for i in range(n):
        u=使d[u]最小的顶点，且未被访问
        vis[u] = true
        for each vertex v in graph[u]:
            if vis[u] == False and d[v] > d[u] + w[u][v]:
                d[v] = d[u] + w[u][v]
    return d
```
## Floyd算法
```py
for each vertex v in graph:
    for each vertex u in graph:
        for each vertex w in graph:
            if d[u][v] + d[v][w] < d[u][w]:
                d[u][w] = d[u][v] + d[v][w]
```

# 是否有重复子串
return s in (s+s)[1:-1]

假设母串S是由子串s重复N次而成， 则 S+S则有子串s重复2N次， 那么现在有： S=Ns， S+S=2Ns， 其中N>=2。 如果条件成立， S+S=2Ns, 掐头去尾破坏2个s，S+S中还包含2*（N-1）s, 又因为N>=2, 因此S在(S+S)[1:-1]中必出现一次以上

# 76. 最小覆盖子串
采用类似滑动窗口的思路，即用两个指针表示窗口左端left和右端right。 向右移动right，保证left与right之间的字符串足够包含需要包含的所有字符， 而在保证字符串能够包含所有需要的字符条件下，向右移动left，保证left的位置对应为需要的字符，这样的 窗口才有可能最短，此时只需要判断当期窗口的长度是不是目前来说最短的，决定要不要更新minL和minR（这两个 变量用于记录可能的最短窗口的端点）

搞清楚指针移动的规则之后，我们需要解决几个问题，就是怎么确定当前窗口包含所有需要的字符，以及怎么确定left的 位置对应的是需要的字符。 这里我们用一个字典mem保存目标字符串t中所含字符及其对应的频数。比如t="ABAc",那么字典mem={"A":2,"B":1,"c":1}, 只要我们在向右移动right的时候，碰到t中的一个字符，对应字典的计数就减一，那么当字典这些元素的值都不大于0的时候， 我们的窗口里面就包含了所有需要的字符；但判断字典这些元素的值都不大于0并不能在O(1)时间内实现，因此我们要用一个变量 来记录我们遍历过字符数目，记为t_len，当我们遍历s的时候，碰到字典中存在的字符且对应频数大于0，就说明我们还没有找到 足够的字符，那么就要继续向右移动right，此时t_len-=1；直到t_len变为0，就说明此时已经找到足够的字符保证窗口符合要求了。

所以接下来就是移动left。我们需要移动left，直到找到目标字符串中的字符，同时又希望窗口尽可能短，因此我们就希望找到的 left使得窗口的开头就是要求的字符串中的字符，同时整个窗口含有所有需要的字符数量。注意到，前面我们更新字典的时候， 比如字符"A",如果我们窗口里面有10个A，而目标字符串中有5个A，那此时字典中A对应的计数就是-5，那么我要收缩窗口又要保证 窗口能够包含所需的字符，那么我们就要在收缩窗口的时候增加对应字符在字典的计数，直到我们找到某个位置的字符A，此时字典中 的计数为0，就不可以再收缩了（如果此时继续移动left，那么之后的窗口就不可能含有A这个字符了），此时窗口为可能的最小窗口，比较 更新记录即可。
```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        char_count=defaultdict(int)
        for char in t:
            char_count[char]+=1
        t_len=len(t)  # 统计当前区间包含t中字母的个数
        min_left,min_right=0,len(s)
        left=0
        res=''
        for right,char in enumerate(s):
            if char_count[char]>0:
                t_len-=1
            char_count[char]-=1
            if t_len==0:
                while char_count[s[left]]<0:
                    char_count[s[left]]+=1
                    left+=1
                if right-left<min_right-min_left:
                    min_left,min_right = left,right
                    res=s[min_left:right+1]
                char_count[s[left]]+=1
                t_len+=1
                left+=1
        return res
```

# rand 问题
(randX() - 1)*Y + randY() 可以等概率的生成[1, X * Y]范围的随机数

# 最长上升子序列
```python
#
# retrun the longest increasing subsequence
# @param arr int整型一维数组 the array
# @return int整型一维数组
#
class Solution:
    def LIS(self , arr ):
        
        # 1. 动态规划，超时
        if len(arr) < 2:
            return arr
        
        dp = [1] * len(arr)
        for i in range(1, len(arr)):
            for j in range(i):
                if arr[i] > arr[j]:
                    dp[i] = dp[j] + 1
        
        ansLen = max(dp)
        ansVec = []
        for i in range(len(arr)-1, -1, -1):
            if dp[i] == ansLen:
                ansVec.insert(0, arr[i])
                ansLen -= 1
                
        return ansVec
        
        
        # 2. 贪心 + 二分
        if len(arr) < 2:
            return arr
        
        ansVec = [arr[0]] # 记录以某一元素结尾的最长递增子序列，初始化为数组第一位元素
        maxLen = [1] # 记录下标i处最长递增子序列的长度，初始化为[1](下标0此时只有一个数字，长度为1)
        
        for num in arr[1:]:
            if num > ansVec[-1]:
                ansVec.append(num)  # 更新以num为结尾元素的最长递增子序列
                maxLen.append(len(ansVec))  # 同时更新此时最长递增子序列的长度
            else:
                """
                for i in range(len(ansVec)):
                    if ansVec[i] >= num:
                        ansVec[i] = num
                        maxLen.append(i+1)
                        break
                """
                # 二分查找第一个比num大的数字，替换
                # 此时以该元素为结尾的最长递增子序列的长度为其在ansVec中的下标+1
                left, right = 0, len(ansVec)-1
                while left < right:
                    mid = (left + right) // 2
                    if ansVec[mid] < num:
                        left = mid+1
                    elif ansVec[mid] == num:
                        left = mid
                        break
                    else:
                        if ansVec[mid-1] < num:
                            left = mid
                            break
                        else:
                            right = mid-1
                ansVec[left] = num
                maxLen.append(left+1)
        
        # ansVec不一定是最后结果，求解按字典序最小的结果
        # 此时我们知道最长长度为ansLen，从后向前遍历maxLen，
        # 遇到第一个maxLen[i]==ansLen的下标i处元素arr[i]即为所求，
        # 例：
        # [1,2,8,6,4] -> maxLen [1,2,3,3,3] -> ansLen=3
        # [1,2,8] -> [1,2,6] -> [1,2,4] 长度一致的答案，字典序越小越靠后
        ansLen = len(ansVec)
        for i in range(len(arr)-1, -1, -1):
            if maxLen[i] == ansLen:
                ansVec[ansLen-1] = arr[i]
                ansLen -= 1
                
        return ansVec
```
# 找环形链表的入口点

首先，可以使用快慢指针找环

如果存在环，那么快慢指针会在环内相遇，但是相遇的点，不一定是环的起点，慢指针走了（a+b）的长度

将快指针放回head，快慢指针以相同的步进移动，最终，会在环的入口相遇，慢指针再走（a+b）的长度会回到原点，只走a的长度会回到环的入口，因此再从head同时和慢指针开始走就行了

你吹过我来时的风，我回到故乡，竟再次在原点相遇
```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        fast = head
        slow = head
        while fast != None and fast.next != None:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                fast = head
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return slow
        return None
```
# 两数之和
```python
#先排序再双指针
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        sorted_id = sorted(range(len(nums)), key=lambda k: nums[k])
        head = 0
        tail = len(nums) - 1
        sum_result = nums[sorted_id[head]] + nums[sorted_id[tail]]
        while sum_result != target:
            if sum_result > target:
                tail -= 1
            elif sum_result < target:
                head += 1
            sum_result = nums[sorted_id[head]] + nums[sorted_id[tail]]
        return [sorted_id[head], sorted_id[tail]]
#字典查找
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for index, num in enumerate(nums):
            another_num = target - num
            if another_num in hashmap:
                return [hashmap[another_num], index]
            hashmap[num] = index
        return None
```
# 接雨水
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        left=[0 for _ in range(len(height))]
        right=[0 for _ in range(len(height))]
        for i in range(1,len(height)):
            left[i]=max(left[i-1],height[i-1])
        for i in range(len(height)-2,-1,-1):
            right[i]=max(right[i+1],height[i+1])
        res=0
        for i in range(len(height)):
            level=min(left[i],right[i])
            res+=max(0,level-height[i])
        return res

# 首先，把h1和h2的面积用标记起来，然后你就会发现：

# h1+h2=2*(柱子面积+水面积)+h1和h2不重叠区域面积

# 因为 柱子面积+水面积+h1和h2不重叠区域面积 = 矩形面积(最高柱子高度*数组长度)

# 所以 h1+h2 = 矩形面积面积+柱子面积+水面积

# 所以 水面积 = h1+h2-矩形面积-柱子面积

# 倒数第二行代码 -height[i] 其实就是减去柱子的面积，所以循环结束后，再减去矩形面积就是最后结果
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        h1 = 0
        h2 = 0
        for i in range(len(height)):
            h1 = max(h1,height[i])
            h2 = max(h2,height[-i-1])
            ans = ans + h1 + h2 -height[i]
        return  ans - len(height)*h1
```