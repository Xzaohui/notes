# python
## deque 双向队列 
appendleft,extendleft

## itertools.groupby
groupby支持两个参数，第一个参数是需要迭代的对象，第二个函数key代表分组依据，如果为none则表示使用迭代对象中的元素作为分组依据

[(ch, list(seq)) for ch, seq in groupby(text)]
## collections.Counter
该方法用于统计某序列中每个元素出现的次数，以键值对的方式存在字典中。

max(counts.keys(),key=lambda x:counts[x])

max(counts.keys(),key=counts.get)

返回统计最多的单词
## all
all(） 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 True，否则返回 False。
## max/min
max/min其实是按照元素里面的第一个元素的排列顺序，输出最大值。如果第一个元素相同，则比较第二个元素，输出最大值。

max([max(d) for d in dp])

max(map(max, dp))
## 反转链表
充分利用python可以多对多赋值的特性
```python
def reverse(head):
    pre, cur = None, head
    while cur:
        cur.next, pre, cur = pre, cur, cur.next
    return pre
```

## 堆 heapq
```python
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
## re match、search、findall、finditer

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
使用 bisect 模块的方法之前，要求操作对象是 有序序列（升序，降序取负号）

bisect 与 bisect_left，insort_left 的区别：当插入的元素和序列中的某一个元素相同时，该插入到该元素的前面（左边，left），还是后面（右边）；如果是查找，则返回该元素的位置还是该元素之后的位置。

查找： bisect.bisect/bisect_left/bisect_right(array, item)

插入： bisect.insort/insort_left/insort_right(array,item)
```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        q = []
        res = 0
        for v in nums:
            i = bisect.bisect_left(q,-v) #bisect后的插入保证了有序性，同时可以反应出有几个比目标数大。
            res += i
            q[i:i] = [-v] #切片做插入，速度快很多
        return res
```
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
## zip/itertools.zip_longest
```python
zip(iterable1, iterable2, ...)
#54. 螺旋矩阵
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if len(matrix)==1:
            return list(matrix[0])
        return list(matrix[0])+list(self.spiralOrder(list(zip(*(matrix[1:])))[::-1]))

# 48. 旋转图像矩阵
matrix[:] = zip(*matrix[::-1])
```

itertools.zip_longest和zip作用基本相同，但是多了一个fillvalue参数，可以指定填充的值。

165. 比较版本号
```python
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        for x, y in zip_longest(version1.split('.'), version2.split('.'), fillvalue='0'):
            a, b = int(x), int(y)
            if a != b: return 1 if a > b else -1
        return 0 
```
## 类的运算符重载

比较运算符（<，<=，>，> =，==和！=）可以通过为__ lt __ ，__ le __ ，__ gt __ ，__ ge __ ，__ eq__和__ne__魔术方法提供定义来重载，以比较类的对象。 

```python
def __lt__(self, other):
    return self.val < other.val
ListNode.__lt__ = __lt__
```
让类也能排序做对比建堆 https://leetcode.cn/problems/merge-k-sorted-lists/

# 树

## 前中后序

### 递归
访问和递归项的位置

### 迭代

借助栈，先将根节点放入栈中，然后将右孩子加入栈，再加入左孩子。再pop一个后一样将右孩子加入栈，再加入左孩子

中续先左树入栈，pop后右树入栈

**后序遍历，先序遍历是中左右，后续遍历是左右中，那么我们只需要调整一下先序遍历的代码顺序，就变成中右左的遍历顺序，然后在反转result数组，输出的结果顺序就是左右中了**
    
```python
# 前序遍历-迭代-LC144_二叉树的前序遍历
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        # 根结点为空则返回空列表
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            # 中结点先处理
            result.append(node.val)
            # 右孩子先入栈
            if node.right:
                stack.append(node.right)
            # 左孩子后入栈
            if node.left:
                stack.append(node.left)
        return result
        
# 中序遍历-迭代-LC94_二叉树的中序遍历
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = []  # 不能提前将root结点加入stack中
        result = []
        cur = root
        while cur or stack:
            # 先迭代访问最底层的左子树结点
            if cur:     
                stack.append(cur)
                cur = cur.left		
            # 到达最左结点后处理栈顶结点    
            else:		
                cur = stack.pop()
                result.append(cur.val)
                # 取栈顶元素右结点
                cur = cur.right	
        return result
        
# 后序遍历-迭代-LC145_二叉树的后序遍历
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            # 中结点先处理
            result.append(node.val)
            # 左孩子先入栈
            if node.left:
                stack.append(node.left)
            # 右孩子后入栈
            if node.right:
                stack.append(node.right)
        # 将最终的数组翻转
        return result[::-1]
```
## 层序遍历
```python
# 队列 bfs
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root==None:
            return []
        q=deque()
        q.appendleft(root)
        ans=[]
        ans.append([root.val])
        while True:
            t=[]
            tq=deque()
            while len(q)>0:
                node=q.popleft()
                if node.left!=None:
                    t.append(node.left.val)
                    tq.append(node.left)
                if node.right!=None:
                    t.append(node.right.val)
                    tq.append(node.right)
            if len(t)>0:
                ans.append(t)
            q=tq
            if len(tq)==0:
                break
        return ans
# 前序遍历 dfs
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        ans=[[]]
        if root==None:
            return []
        def dfs(root,deep,ans):
            if root is None:
                return
            if deep>len(ans)-1:
                ans.append([])
            ans[deep].append(root.val)
            dfs(root.left,deep+1,ans)
            dfs(root.right,deep+1,ans)
        dfs(root,0,ans)
        return ans
```

## 二叉树
二叉树共父节点 https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/comments/
```python
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
```python
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
            while cur.left: cur = cur.left #右子树的最小值，也可用左子树的最大值和要删除的做替换
            root.val = cur.val
            root.right = deleteNode(root.right, cur.val) 
    return root
```
二叉搜索树是否合法
```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def valid(node,min,max):
            if  not node:
                return True
            if node.val<=min or node.val>=max:
                return False
            return valid(node.left,min,node.val) and valid(node.right,node.val,max)
        return valid(root,-inf,inf)
```


## 平衡二叉树

```python
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


## 124. 二叉树中的最大路径和
https://leetcode.cn/problems/binary-tree-maximum-path-sum/
```python
class Solution:
    
    # 对于任意一个节点, 如果最大和路径包含该节点, 那么只可能是两种情况:
    # 1. 其左右子树中所构成的和路径值较大的那个加上该节点的值后向父节点回溯构成最大路径
    # 2. 左右子树都在最大路径中, 加上该节点的值构成了最终的最大路径

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        val=-99999
        def getMax(root):
            nonlocal val
            if root is None:
                return 0
            right=max(0,getMax(root.right)) #如果子树路径和为负则应当置0表示最大路径不包含子树
            left=max(0,getMax(root.left))
            val=max(val,root.val+left+right) #判断在该节点包含左右子树的路径和是否大于当前最大路径和
            return max(left,right)+root.val  #回溯往上找最大值
        getMax(root)
        return val
```

## 572. 另一棵树的子树
https://leetcode.cn/problems/subtree-of-another-tree/submissions/

遍历同时检查是否相同
```python
class Solution:
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        def sametree(root,subroot):
            if root != None and subroot!=None:
                return root.val==subroot.val and sametree(root.left,subroot.left) and sametree(root.right,subroot.right)
            elif root == None and subroot==None:
                return True
            else:
                return False
        if root==None:
            return False
        return sametree(root,subRoot) or self.isSubtree(root.left,subRoot) or self.isSubtree(root.right,subRoot)
```

## 101. 对称二叉树
```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        return self.compare(root.left, root.right)
            
    def compare(self, left, right):
        #首先排除空节点的情况
        if left == None and right != None: return False
        elif left != None and right == None: return False
        elif left == None and right == None: return True
        #排除了空节点，再排除数值不相同的情况
        elif left.val != right.val: return False
        
        #此时就是：左右节点都不为空，且数值相同的情况
        #此时才做递归，做下一层的判断
        outside = self.compare(left.left, right.right) #左子树：左、 右子树：右
        inside = self.compare(left.right, right.left) #左子树：右、 右子树：左
        isSame = outside and inside #左子树：中、 右子树：中 （逻辑处理）
        return isSame
```

## 公祖问题

```python
def lowestCommonAncestor(root, p, q):
    if root is None or root == p or root == q: #如果root是p或者q，那么直接返回，若p或q是公祖则后面也不用找了
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:      # 如果左右子树都不为空，说明p和q分别在左子树和右子树上，那么这个节点就是公共祖先
        return root
    return left if left else right # 如果左右子树有一个为空，说明p和q在同一侧，返回前一侧的子树

非递归版本
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        a = []
        path = [root]
        def dfs(root):
            if not root: return
            if root.val in [p.val,q.val]:
                a.append(list(path))    # 记录路径
            for node in [root.left,root.right]:
                path.append(node)
                dfs(node)
                path.pop()
        dfs(root)
        i=0
        while i<min(len(a[0]),len(a[1])) and a[0][i].val==a[1][i].val:
            i+=1
        return a[0][i-1]
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
    
    通过二分查找的方法，快速查找到某个字符应该匹配的子节点的指针。（这就不用维护一个上述26的数组，只需要维护两个可能的字符数组）当然，这样为了维护数组顺序，插入元素效率较慢。

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


## 440. 字典序的第K小数字
```python
# 本质是一个10叉树的先序遍历,找到按照先序遍历的第k个节点
# 为什么是先序遍历?这个由字典序的性质决定:[1,10,100,1000,1001]
# 假设相同位数的数字在10叉树的同一层上,那么就是先序遍历就是字典序排列
# 从cur=1开始进行遍历,先计算的以cur为根的且<=n的节点个数nodes
# 若nodes<=k,说明以cur开头的合格节点数不够,cur应该向右走:cur++
# 若nodes>k,说明以cur开头的合格节点数足够,cur应该向下走:cur*=10
class Solution:
    def findKthNumber(self, n: int, k: int) -> int:
        cur=1
        k-=1 #1已经计算过了
        def count_tree(cur): #计算以cur为根的且<=n的节点个数
            next=cur+1 #相邻的下一个节点
            num=0
            while cur<=n:
                #这里是最关键的一步:当n不在cur层时,该层有效节点数目为next - cur(全部都要了)
                #当n在cur层时,该层有效节点数目为n - cur + 1(要一部分)
                #统一起来就是取最小值
                num+=min(n-cur+1,next-cur) 
                cur*=10 #向下走
                next*=10 
            return num
        while k>0:
            ctree=count_tree(cur) 
            if ctree<=k: #以cur开头的合格节点数不够,cur应该向右走
                cur+=1
                k-=ctree #k减去以cur开头的合格节点数
            else:
                cur*=10 #以cur开头的合格节点数足够,cur应该向下走
                k-=1 #cur已经计算过了，k-=1
        
        return cur
```

# 贪心

## 分发糖果

评分高的孩子获得更多的糖果。

一次是从左到右遍历，只比较右边孩子评分比左边大的情况。

一次是从右到左遍历，只比较左边孩子评分比右边大的情况。

```python
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
```python
# 变换cost相同
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
# 变换cost不同，注意由谁变成谁
class Solution:
    def minDistance(self , str1: str, str2: str, ic: int, dc: int, rc: int) -> int:
        dp=[[0]*(len(str1)+1) for _ in range(len(str2)+1)]
        cost=[ic,dc,rc]
        for i in range(len(str1)+1):
            dp[0][i]=i*cost[1]
        for i in range(len(str2)+1):
            dp[i][0]=i*cost[0]

        for i in range(1,len(str1)+1):
            for j in range(1,len(str2)+1):
                if str1[i-1]==str2[j-1]:
                    dp[j][i]=dp[j-1][i-1]
                else:
                    dp[j][i]=min(dp[j-1][i-1]+cost[2],dp[j][i-1]+cost[1],dp[j-1][i]+cost[0])
        return dp[-1][-1]
```
## 115. 不同的子序列
```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        dp=[[0]*(len(s)+1) for _ in range(len(t)+1)]
        for i in range(len(s)+1):
            dp[0][i]=1
        for i in range(1,len(t)+1):
            for j in range(1,len(s)+1):
                if s[j-1]==t[i-1]:
                    dp[i][j]=dp[i][j-1]+dp[i-1][j-1] #字母匹配上了，则前一段匹配的数量和上一次的相加
                else:
                    dp[i][j]=dp[i][j-1] #横向扩展没匹配上字母，组合数和前一次相同
        return dp[-1][-1]
```
## 跳表问题

45. 跳跃游戏 II
```python
for i in range(len(nums)-2,-1,-1):
    jump[i]=min([jump[j] for j in range(i+1,min(i+nums[i]+1,len(nums)))])+1
return jump[0]
```
## 买卖股票
### 只买卖一次
```python
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

```python
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
```python
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

目标和，有+-符号组合：转换为letf-right=traget，left+right=sum，left=(sum+target)/2。再转化为01背包。

一和零：二维01背包问题，两个容量限制。注意两个都要倒序计算。
```python
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

518.零钱兑换问题，518.组合数，没有顺序要求。

377.组合总和有顺序要求。

有次数要求就二维数组，或一维数组从后到前更新。
```python
for coin in coins: #零钱兑换问题
    for i in range(1,amount+1):
        if coin <=i:
            dp[i]+=dp[i-coin]

for i in range(1,target+1):#组合总和
    for num in nums:
        if num <=i:
            dp[i]+=dp[i-num]

return dp[-1]


class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        dp=[[0]*(target+1) for _ in range(n)]
        M = 1000000007
        for i in range(min(k+1,target+1)):
            dp[0][i]=1
        # for j in range(target,0,-1):
        #     dp[j]=0
        #     for m in range(1,min(j,k+1)):
        #         dp[j]+=dp[j-m]    一维dp数组。
        for i in range(1,n):
            for j in range(1,target+1):
                for m in range(1,min(j,k+1)): # 不能投和j一样大的数，影响第二维结果
                    dp[i][j]+=dp[i-1][j-m]
                    dp[i][j]%=M
        return dp[-1][-1]%M
```


### 多重背包问题
有N种物品和一个容量为V 的背包。第i种物品最多有Mi件可用，每件耗费的空间是Ci ，价值是Wi 。求解将哪些物品装入背包可使这些物品的耗费的空间 总和不超过背包容量，且价值总和最大。

转化成01背包，把每个物品扩展Mi次。

## 成环就考虑两种，一不取头，二不取尾

## 树形后序遍历 dp
337. 打家劫舍 III

https://leetcode.cn/problems/house-robber-iii/
```python
def trob(root):
   if root == None:
       return [0,0]
   r=trob(root.right)
   l=trob(root.left)
   return [root.val+r[1]+l[1],max(r)+max(l)] #不抢也要返回他的max
return max(trob(root))
```
## 矩阵中的最长递增路径
dp[i][j]表示以matrix[i][j]结尾的最长递增长度

初始dp[i][j]都等于1，若matrix[i][j]四个方向有任意小于它，则可以更新dp[i][j] = max(dp[i][j], 1 + dp[r][c])

我们在计算dp[i][j]之前，必须先把dp[r][c]计算出来。 matrix(r,c)位置的值比matrix（i，j）位置的值小（从if语句看出来的）。所以我们只要保证，先把matrix中值小的位置的dp先算出来，再把值大的位置的dp算出来，倒数第二行的代码就有意义了。所以我们要先排序。
```python
#dp
class Solution(object):
    def longestIncreasingPath(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        m, n = len(matrix), len(matrix[0])
        lst = []
        for i in range(m):
            for j in range(n):
                lst.append((matrix[i][j], i, j))
        lst.sort()
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for num, i, j in lst:
            dp[i][j] = 1
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = i + di, j + dj
                if 0 <= r < m and 0 <= c < n:
                    if matrix[i][j] > matrix[r][c]:
                        dp[i][j] = max(dp[i][j], 1 + dp[r][c])
        return max([dp[i][j] for i in range(m) for j in range(n)])

#dfs
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]: return 0

        row = len(matrix)
        col = len(matrix[0])
        lookup = [[0] * col for _ in range(row)]

        def dfs(i, j):
            if lookup[i][j] != 0:
                return lookup[i][j]
            # 方法一
            res = 1
            for x, y in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
                tmp_i = x + i
                tmp_j = y + j
                if 0 <= tmp_i < row and 0 <= tmp_j < col and \
                        matrix[tmp_i][tmp_j] > matrix[i][j]:
                    res = max(res, 1 + dfs(tmp_i, tmp_j))
            lookup[i][j] = max(res, lookup[i][j])
            return lookup[i][j]

        return max(dfs(i, j) for i in range(row) for j in range(col))
```
## 887. 鸡蛋掉落
状态可以表示成 (k,n)，其中 k 为鸡蛋数，n 为楼层数。当我们从第 x 楼扔鸡蛋的时候：

如果鸡蛋不碎，那么状态变成(k,n−x)，即我们鸡蛋的数目不变，但答案只可能在上方的 n-x 层楼了。也就是说，我们把原问题缩小成了一个规模为(k,n−x) 的子问题；

如果鸡蛋碎了，那么状态变成 (k−1,x−1)，即我们少了一个鸡蛋，但我们知道答案只可能在第 xx 楼下方的 x-1x−1 层楼中了。也就是说，我们把原问题缩小成了一个规模为 (k-1, x-1)的子问题。



每一步都应该在第 dp[k-1][t-1] + 1 层丢鸡蛋。

第一,如果蛋碎了,那么我们一定能用k-1个鸡蛋用m-1步测出下面的dp[k-1][m-1]层楼。

第二,如果蛋没碎,最多可以测出上面的dp[k][m-1]层楼(能测出的层数对于上下来说一样)

dp[k][m]那么总共可以测出dp[k-1][m-1]+dp[k][m-1]+1层楼
```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        if n == 1:
            return 1
        f = [[0] * (k + 1) for _ in range(n + 1)]
        for i in range(1, k + 1):
            f[1][i] = 1
        ans = -1
        for i in range(2, n + 1):
            for j in range(1, k + 1):
                f[i][j] = 1 + f[i - 1][j - 1] + f[i - 1][j]
            if f[i][k] >= n:
                ans = i
                break
        return ans


#传统想法
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        memo = {}
        def dp(k, n):
            if (k, n) not in memo:
                if n == 0:
                    ans = 0
                elif k == 1:
                    ans = n
                else:
                    lo, hi = 1, n
                    # keep a gap of 2 x values to manually check later
                    while lo + 1 < hi:
                        x = (lo + hi) // 2
                        t1 = dp(k - 1, x - 1)
                        t2 = dp(k, n - x)

                        if t1 < t2:
                            lo = x
                        elif t1 > t2:
                            hi = x
                        else:
                            lo = hi = x
                    ans = 1 + min(max(dp(k - 1, x - 1), dp(k, n - x))
                                  for x in (lo, hi))
                memo[k, n] = ans
            return memo[k, n]

        return dp(k, n)
```

# 最短路径问题

## dijkstra 算法
```python
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

# bfs
## 127. 单词接龙
```python
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
     search_list=[]
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
```python
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

## 丑数 bfs思想
把只包含因子 2、3 和 5 的数称作丑数（Ugly Number）。
```python
# dp 
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp=[1]
        ind=[0 for _ in range(3)]
        unums=[2,3,5]
        while True:
            if len(dp)>=n:
                break
            t=[dp[ind[j]]*unums[j] for j in range(3)]
            m=min(t)
            if m not in dp:
                dp.append(m)
            ind[t.index(m)]+=1  # 当前队列往下走一个位置
        return dp[-1]
        
# 用堆，出来一个进去3个相应的乘积，判断重复的时候不要用in。
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        heap=[1]
        res=[]
        unums=[2,3,4]
        while len(res)<n:
            temp=heapq.heappop(heap)
            while len(heap)>0 and temp==heap[0]:
                heapq.heappop(heap)
            res.append(temp)
            for u in unums:
                heapq.heappush(heap,u*temp)
        return res[-1]
```

# 递归/回溯/dfs
https://lyl0724.github.io/2020/01/25/1/ 
## 组合总数（可重复/不可重复，可多次存取/不可多次存取）
39. 组合总和
40. 组合总和 II
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res=[]
        candidates=sorted(candidates)
        def dfs(candidates,nums,target,i):
            if sum(nums)==target:
                res.append(nums[:])
                return
            elif sum(nums)>target:
                return
            for j in range(i,len(candidates)):
                # if j>i and candidates[j]==candidates[j-1]:
                #     continue      去重，回溯前面用过了
                nums.append(candidates[j]) 
                dfs(candidates,nums,target,j) #可以重复使用就可以用j，不能重复就j+1
                nums.pop(-1)
        dfs(candidates,[],target,0)
        return res
```
## 排列总数
46. 全排列

交换的思想，也可以直接递归/回溯暴力存取数字
```python
class Solution: #交换
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def permute(nums,i):
            if i == len(nums)-1:
                res.append(nums[:])
            for j in range(i,len(nums)):
                nums[i],nums[j]=nums[j],nums[i]
                permute(nums,i+1)
                nums[i],nums[j]=nums[j],nums[i]
        permute(nums,0)
        return res

class Solution: #暴力递归回溯
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(nums, tmp):
            if not nums:
                res.append(tmp)
                return 
            for i in range(len(nums)):
                backtrack(nums[:i] + nums[i+1:], tmp + [nums[i]])
        backtrack(nums, [])
        return res
```
## 括号生成
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
全排列+剪枝
https://leetcode.cn/problems/generate-parentheses/
## 电话号码
```python
def letterCombinations(self, digits: str) -> List[str]:
        dic={2:"abc",3:"def",4:"ghi",5:"jkl",6:"mno",7:"pqrs",8:"tuv",9:"wxyz"}
        if len(digits)==0:
            return []
        if len(digits)==1:
            return list(dic[int(digits)])
        else:
            return [i+j for i in self.letterCombinations(digits[0]) for j in self.letterCombinations(digits[1:])]

```
https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

## 82. 删除排序链表中的重复元素 II
```python
class Solution(object):
    def deleteDuplicates(self, head):
        if head is None or head.next is None: return head
        if head.val == head.next.val:
            while head.next and head.val == head.next.val:
                head = head.next
            head = self.deleteDuplicates(head.next)
        else:
            head.next = self.deleteDuplicates(head.next)
        return head
```

# 智力题
## N个小球里找次品，天平最少秤几次
情况1：次品的轻重已知

已知次品轻重的情况下，N 个小球一共有 N 种次品可能性，即 N 个小球都可能是次品（比其他小球重）。天平每秤一次，会有三种情况：即左重、右重、或平衡。若每次秤的时候，都能将小球平均分为 3 份，则秤一次天平会将可能性变为之前的 1/3 。

情况2：次品的轻重未知

若次品的轻重未知，则 N 个小球中次品的情况有 2N 中，即每一个小球都有可能成为次品，且该次品较轻或者较重。经过 1 次称重后，可能性也会降低到之前的1/3。最多需要每次多称重一次。

## 平均要抛多少次硬币，才能出现连续K次正面向上？
令正面朝上的概率为p，则有：
$$
E(N_k)=E(N_{k-1})+p+(1-p)(E(N_{k})+1)\\
E(N_k)=\frac{1}{p}+\frac{E_{k-1}}{p}\\ \\

E(N_k)=\frac{1}{p}+\frac{1}{p^2}+\dots+\frac{1}{p^{k-1}}+\frac{1}{p^{k}}\\
$$
## n双鞋，能随机匹配每一双的概率
P(n,n)*2^n/P(2n,2n)
## 是否有重复子串
return s in (s+s)[1:-1]

假设母串S是由子串s重复N次而成， 则 S+S则有子串s重复2N次， 那么现在有： S=Ns， S+S=2Ns， 其中N>=2。 如果条件成立， S+S=2Ns, 掐头去尾破坏2个s，S+S中还包含2*（N-1）s, 又因为N>=2, 因此S在(S+S)[1:-1]中必出现一次以上
## rand 问题
(randX() - 1)*Y + randY() 可以等概率的生成[1, X * Y]范围的随机数

# 分治
## 寻找两个正序数组的中位数
4. 寻找两个正序数组的中位数

这个题目可以归结到寻找第k小(大)元素问题，思路可以总结如下：取两个数组中的第k/2个元素进行比较，如果数组1的元素小于数组2的元素，则说明数组1中的前k/2个元素不可能成为第k个元素的候选，所以将数组1中的前k/2个元素去掉，组成新数组和数组2求第k-k/2小的元素，因为我们把前k/2个元素去掉了，所以相应的k值也应该减小。另外就是注意处理一些边界条件问题，比如某一个数组可能为空或者k为1的情况。

我们分别找第 (m+n+1) / 2 个，和 (m+n+2) / 2 个，然后求其平均值即可，这对奇偶数均适用。
```python
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def findKthElement(arr1,arr2,k): # 找第k小/大的元素
            len1,len2 = len(arr1),len(arr2)
            if not arr1:
                return arr2[k-1]
            if not arr2:
                return arr1[k-1]
            if k == 1:
                return min(arr1[0],arr2[0])
            i,j = min(k//2,len1)-1,min(k//2,len2)-1 # 每次取k/2处的两个进行比较
            if arr1[i] > arr2[j]:
                return findKthElement(arr1,arr2[j+1:],k-j-1) # k要减去抛弃的数组的长度
            else:
                return findKthElement(arr1[i+1:],arr2,k-i-1)
        l1,l2 = len(nums1),len(nums2)
        left,right = (l1+l2+1)//2,(l1+l2+2)//2 # 一次性包括了奇偶两种情况
        return (findKthElement(nums1,nums2,left)+findKthElement(nums1,nums2,right))/2
```
https://leetcode.cn/problems/median-of-two-sorted-arrays/

## 合并k个排序的链表
23. 合并K个升序链表

分治+递归合并，递归合并两个链表，分治分别合并。
```python
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

https://leetcode.cn/problems/merge-k-sorted-lists/

## 寻找峰值
规律一：如果nums[i] > nums[i+1]，则在i之前一定存在峰值元素

规律二：如果nums[i] < nums[i+1]，则在i+1之后一定存在峰值元素

导数异号且连续必有峰值
```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left,right=0,len(nums)-1
        while left<right:
            mid=(left+right)//2
            if nums[mid]>nums[mid+1]:
                right=mid
            else:
                left=mid+1
        return left
```


## 最长上升子序列 二分/dp
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
                if arr[i] > arr[j]: #依次找比前面大的数，再加上去后和现在最长相比
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

# 链表
## 链表的倒数第 N 个结点
先让快指针走N步，再快慢一起走，快走到头则慢在倒数第N个结点处

## 链表排序
```python
# 快排思想，把小于基准值的放在左边，大于基准值的放在右边，基准值放在中间
def sortList(head):
    pre=ListNode()
    pre.next=head
    def sort(head,end):
        if head==None or  head.next==end or head.next.next==end:
            return head
        temp=ListNode()
        temp1=temp
        p=head
        temp1.next=p.next
        p.next=p.next.next
        temp1=temp1.next
        temp1.next=None
        while p.next!=end:
            if p.next.val<temp1.val:
                t,temp.next,p.next=temp.next,p.next,p.next.next
                temp.next.next=t
            else:
                p=p.next
        temp1.next,head.next=head.next,temp.next
        sort(head,temp1)
        sort(temp1,end)
        return head.next
    sort(pre,None)
    return pre.next
```
## 找环形链表的入口点

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

# 滑动窗口
## 76. 最小覆盖子串
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
## 滑动窗口最大值
```python
# 维护窗口，向右移动时左侧超出窗口的值弹出，因为需要的是窗口内的最大值，所以只要保证窗口内的值是递减的即可，小于新加入的值全部弹出。最左端即为窗口最大值 
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        win, ret = [], []
        for i, v in enumerate(nums):
            if i >= k and win[0] <= i - k: win.pop(0)
            while win and nums[win[-1]] <= v: win.pop()
            win.append(i)
            if i >= k - 1: ret.append(nums[win[0]]) # 只有走到窗口边缘时才能加入ret
        return ret
```

# 双指针
## 两数之和
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

3数之和也可以用双指针，先固定一个再当成2数

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

# 最长回文子序列
```cpp
#双指针法
class Solution {
public:
    int left = 0;
    int right = 0;
    int maxLength = 0;
    string longestPalindrome(string s) {
        int result = 0;
        for (int i = 0; i < s.size(); i++) {
            extend(s, i, i, s.size()); // 以i为中心
            extend(s, i, i + 1, s.size()); // 以i和i+1为中心
        }
        return s.substr(left, maxLength);
    }
    void extend(const string& s, int i, int j, int n) {
        while (i >= 0 && j < n && s[i] == s[j]) {
            if (j - i + 1 > maxLength) {
                left = i;
                right = j;
                maxLength = j - i + 1;
            }
            i--;
            j++;
        }
    }
};
#动态规划
class Solution {
public:
    string longestPalindrome(string s) {
        vector<vector<int>> dp(s.size(), vector<int>(s.size(), 0));
        int maxlenth = 0;
        int left = 0;
        int right = 0;
        for (int i = s.size() - 1; i >= 0; i--) {
            for (int j = i; j < s.size(); j++) {
                if (s[i] == s[j]) {
                    if (j - i <= 1) { // 情况一 和 情况二
                        dp[i][j] = true;
                    } else if (dp[i + 1][j - 1]) { // 情况三
                        dp[i][j] = true;
                    }
                }
                if (dp[i][j] && j - i + 1 > maxlenth) {
                    maxlenth = j - i + 1;
                    left = i;
                    right = j;
                }
            }

        }
        return s.substr(left, right - left + 1);
    }
};
```
# 最长公共前缀
```python
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
```python
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

# 下一个排列
最后一个数字往前找一个比他小的，交换后，后面的再排序
```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(len(nums)-1,-1,-1):
            for j in range(len(nums)-1,i,-1):
                if nums[i]<nums[j]:
                    nums[i],nums[j]=nums[j],nums[i]
                    nums[i+1:]=sorted(nums[i+1:])
                    return
            
        nums.reverse()
```


# KMP

```python
# next数组
def cal_next(ptr):
	next = [-1]
    # ptr第一位的没有最长前后缀，直接赋值为-1
	k = -1
    # k代表最长前后缀长度，赋值为-1
	for p in range(1, len(ptr)):
        # 从第二位开始遍历ptr
		while k>-1 and ptr[p]!=ptr[k+1]:
            # 假设已有最长前缀为A，最长后缀为B，B的下一位ptr[p] != A的下一位ptr[k+1]
            # 说明最长前后缀不能持续下去
			k = next[k]
            # 往前回溯，尝试部分前缀，而非从第一位开始重新寻找最长前缀
		if ptr[p] == ptr[k+1]:
            # 如果A B的下一位相同
			k = k + 1
            # 最长前后缀长度 + 1
		next.append(k)
            # 第p位的最长前后缀赋值为k
	print('next: ', next)
	return next
# 匹配
def kmp(str, ptr):
	next = cal_next(ptr)    # 求解next
	k = -1    # 此处k相当于ptr中已匹配的长度，类似一个指针指向ptr中已匹配的最后一位
	num = 0    # str中ptr的数量
	for p in range(len(str)):
        # 遍历str
		while k>-1 and str[p] != ptr[k+1]:
            # 假设str中的A片段和ptr中的前A位已匹配，但A的下一位和ptr中A+1位不匹配
			k = next[k]
            # 放弃A片段中的前若干位，因为它们不可能再匹配了
            # 用A片段的后若干位去匹配ptr中某一个最大前缀，像上面矩形图所示
		if str[p] == ptr[k+1]:
            # 如果A的下一位和ptr中A+1位相匹配
			k = k+1
		if k == len(ptr)-1:    # 如果ptr走到尽头
			num = num + 1    # 匹配到了一个
			k = next[k]
	return num
```

# 开方
```python
# 二分法
def mySqrt(x: int) -> int:
    left,right=0,x
    while left<=right:
        mid=(left+right)/2
        if mid*mid>x:
            right=mid-0.00001 #控制精度，+=1精度就是整数
        elif mid*mid<x:
            left=mid+0.00001
        else:
            return mid
    print(right)
    return right

# 牛顿法，求迭代式
class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        
        C, x0 = float(x), float(x)
        while True:
            xi = 0.5 * (x0 + C / x0)
            if abs(x0 - xi) < 1e-7:
                break
            x0 = xi
        
        return int(x0)
```
# 232. 用栈实现队列
两个栈。一个负责出一个负责入