# python

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
## match、search、findall、finditer
都可以用r''正则表达式来匹配

match方法从头开始找，找到就返回，否则为None，只匹配一次（必须开头就有这个字符串）

search从头依次搜索，只匹配一次

findall方法：返回列表，匹配所有

返回string中所有相匹配的全部字串，返回形式为迭代器。

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


# 递归
https://lyl0724.github.io/2020/01/25/1/ 

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

# deque 双向队列 

appendleft,extendleft

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

# bisect
查找： bisect.bisect/bisect_left/bisect_right(array, item)

插入： bisect.insort/insort_left/insort_right(array,item)

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

## 跳表问题

```py
for i in range(len(nums)-2,-1,-1):
    jump[i]=min([jump[j] for j in range(i+1,min(i+nums[i]+1,len(nums)))])+1
return jump[0]
```
## 买卖股票的最佳时机含手续费
```py
dp1 = [0 for _ in range(n)]#第i天手上有股票时的最大收益
dp2 = [0 for _ in range(n)]#第i天手上无股票时的最大收益
dp1[0] = -prices[0]
for i in range(1,n):
    dp1[i] = max(dp1[i-1], dp2[i-1] - prices[i]) #准备抄底
    dp2[i] = max(dp2[i-1], dp1[i-1] + prices[i] - fee) #见好就收
return max(dp1[n-1], dp2[n-1])
```

