import torch


class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            histogram: dict or tensor. If dict, keys are num_nodes, values are counts
        """
        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram
            
        # 增加最大节点数限制
        if len(prob) < 75:  # 设置更大的安全阈值，比最大节点数多5个以上
            extended_prob = torch.zeros(75)
            extended_prob[:len(prob)] = prob
            prob = extended_prob
            
        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(self.prob)
        self.max_n = len(self.prob) - 1

        
    def sample_n(self, n_samples, device):  # 保持方法名一致
        """从节点分布中采样
        Args:
            n_samples: 需要采样的数量
            device: 设备类型
        Returns:
            采样得到的节点数
        """
        idx = self.m.sample((n_samples,))
        return idx.to(device)        
        
        
        
    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)
        
        # 处理超出范围的节点数
        if torch.any(batch_n_nodes > self.max_n):
            print(f"Warning: some nodes exceed max_n: {batch_n_nodes.max()} > {self.max_n}")
            batch_n_nodes = torch.clamp(batch_n_nodes, max=self.max_n)
            
        probas = p[batch_n_nodes]
        # 添加一个小的常数以避免log(0)
        log_p = torch.log(probas + 1e-30)
        return log_p
