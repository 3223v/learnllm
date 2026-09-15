import numpy as np

# 不纯度：方差，也就是n*Var = np.var(y) * len(y)
# 分裂增益：Gain = Var(parent) - (n_left / n) * Var(left) - (n_right / n) * Var(right)
# 分裂后不纯度的减少
# 叶子输出：回归树输出该叶子内的样本的均值


class Node:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_index = feature_index  # 分裂特征索引
        self.threshold = threshold  # 分裂阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 节点的类别标签（叶子节点）y的均值

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth  # 最大深度
        self.root = None
        self.min_samples_split = min_samples_split  # 最小分裂样本数
        self.min_samples_leaf = min_samples_leaf  # 最小叶子节点样本数

    def fit(self, X, y):
        X = np.asanyarray(X, dtype=float)
        y = np.asanyarray(y, dtype=float)
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        n_samples, n_features = X.shape
        print(f"当前深度: {depth}, 样本数: {n_samples}, 特征数: {n_features}")
        # 停止条件（满足任意一个就生成叶子）
        # 1.达到最大深度  2.样本太少，不能分裂 3.全部y值相同，无需分裂
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            return Node(value=np.mean(y))
        best = self._best_split(X, y)
        if best is None:
            return Node(value=np.mean(y))

        feature, threshold, left_idx, right_idx = best

        left = self._build(X[left_idx], y[left_idx], depth + 1)
        right = self._build(X[right_idx], y[right_idx], depth + 1)
        return Node(feature_index=feature, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_sse = np.var(y) * n_samples  # 父节点的总平方误差
        best_gain = 0
        best_split = None

        for f in range(n_features):
            values = np.unique(X[:, f])
            uniques = np.unique(X[:, f])
            if len(uniques) <= 1:
                continue
            thresholds = (uniques[:-1] + uniques[1:]) / 2  # 计算中间阈值
            for t in thresholds:
                left_mask = X[:, f] <= t
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                left_sse = np.var(y[left_mask]) * n_left
                right_sse = np.var(y[~left_mask]) * n_right
                gain = parent_sse - (left_sse + right_sse)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (f, t, left_mask, ~left_mask)
        return best_split

    def predict(self, X):
        X = np.asanyarray(X, dtype=float)
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)
rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(300, 2))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.1, 300)

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
pred = tree.predict(X)

print("训练 MSE:", np.mean((pred - y) ** 2))