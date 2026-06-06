# REPORT

## *Part 1 : 补全的函数*

### *1. `td_estimate`*

相当于当前网络的打分，实现想法见注释 ：

```
def td_estimate(self, state, action):
        """
        根据 online 网络返回当前 batch 的 Q(s, a)。
        提示：online 网络算 Q 值表，再按 action 取对应分数。见 ASSIGNMENT.md。
        """

        # 1. 将当前状态输入 online 网络，得到所有动作的 Q 值
        current_q_values = self.net(state, model="online")
        # 2. 根据实际执行的 action，提取对应的 Q(s, a) 值
        # np.arange(0, self.batch_size) 遍历每一行，按action进行对应提取
        current_q = current_q_values[np.arange(0, self.batch_size), action]
        return current_q
```

### *2. `td_target`*

想法见注释 ：

```
@torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        根据 DQN 目标公式计算 TD target。
        提示：target 网络算 next_state 最高分，再写 return 公式。见 ASSIGNMENT.md。
        """

        # 1. 使用 target 网络计算下一状态的所有动作 Q 值
        next_q_values = self.net(next_state, model="target")
        # 2. 取下一状态的最大 Q 值 (max Q_target(s', a'))
        next_q = next_q_values.max(dim=1)[0]
        # 3. 根据 DQN 公式计算 TD Target
        # 如果 done 为 True (即 done.float()=1)，则 (1-done)=0，后续 Q 值不计入
        # 如果 done 为 False (即 done.float()=0)，则加上 gamma * next_q
        return (reward + (1 - done.float()) * self.gamma * next_q).float()
```

### *3. `update_Q_online`*

完成参数更新，实现方法见注释：

```
def update_Q_online(self, td_estimate, td_target):
        """
        使用 `self.loss_fn`、`self.optimizer` 完成一次参数更新。
        提示：loss_fn → zero_grad → backward → step → return loss.item()。
        """

        # 1. 计算 TD Estimate 和 TD Target 之间的损失
        loss = self.loss_fn(td_estimate, td_target)
        # 2. 清空过往梯度
        self.optimizer.zero_grad()
        # 3. 反向传播计算梯度
        loss.backward()
        # 4. 更新 online 网络参数
        self.optimizer.step()
        # 5. 返回标量形式的 loss 值，方便后续记录
        return loss.item()
```

## *Part 2 : TD target 计算方式*

依照贝尔曼最优方程：

$$
Y_t = R_{t+1} + \gamma \cdot \max_{a'} Q(S_{t+1}, a'; \theta^-)
$$

如果游戏结束，那么没有未来收益，公式变为：
$$
Y_t = R_{t+1}
$$
我们使用`1-done.float()`来控制后面的项，如果结束，`done.float() = 1`

在这里我们 return 的内容事实上就是贝尔曼最优方程 ：`reward + (1 - done.float()) * self.gamma * next_q`

## *Part3 : 程序运行情况与结果*

### 1. 首先是环境配置：

依据 `environment.yml`，我做了微调：

torch 版本 2.11.0+cu128，torchvision 版本0.26.0+cu128 适配显卡

其余均按照文件版本配置。

### 2. 然后是第一次实验

`python main.py --episodes 100 --gpu 0 `

`episodes = 100`仅仅是个热身

结果：

- 平均存活步数：

![length](checkpoints/2026-06-06T10-43-53/length_plot.jpg)

显然波动明显，早期随即探索不同的动作，纯看运气，而且此时轮次少，并没有很有效的经验

- 平均损失

![loss](checkpoints/2026-06-06T10-43-53/loss_plot.jpg)

前面经验池没有攒够数据，网络还没有开始训练，所以是 0。从60回合开始上升，说明开始训练了。

- 平均Q值

![Q](checkpoints/2026-06-06T10-43-53/q_plot.jpg)

前期为0，后期攀升。上升说明他觉得自己处于某种有价值的状态，但是初期大概率是虚高，很常见

- 累计分数

![reward](checkpoints/2026-06-06T10-43-53/reward_plot.jpg)

分数在600分左右，相当于只是开始的一小段，处于早期。





