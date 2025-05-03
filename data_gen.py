import json
import numpy as np
import pandas as pd
import random
import os
from collections import deque, defaultdict

def extract_features_and_classes(tree):
    """
    从树中提取所有特征和类别
    
    Args:
        tree: 树的BFS表示
        
    Returns:
        features: 所有特征的集合
        classes: 所有类别的集合
    """
    features = set()
    classes = set()
    
    for node in tree:
        if node is None:
            continue
            
        if node["role"] == "C":
            # 从条件节点中提取特征
            for condition in node["triples"]:
                features.add(condition)
        elif node["role"] == "D":  # 移除了 and node["triples"] 条件，现在空决策节点也会被考虑
            # 从决策节点中提取类别
            class_tuple = tuple(sorted(node["triples"]))
            classes.add(class_tuple)
    
    return features, classes

def find_path_to_leaf(tree, leaf_index):
    """
    找到从根节点到叶节点的路径
    
    Args:
        tree: 树的BFS表示
        leaf_index: 叶节点在BFS表示中的索引
        
    Returns:
        path: 从根节点到叶节点的路径，包含每个节点的索引
    """
    path = []
    current_index = leaf_index
    
    # 从叶节点回溯到根节点
    while current_index > 0:
        path.append(current_index)
        # 计算父节点索引
        parent_index = (current_index - 1) // 2
        current_index = parent_index
    
    # 添加根节点
    path.append(0)
    # 反转路径，使其从根节点开始
    path.reverse()
    
    return path

def get_leaf_nodes_by_class(tree):
    """
    按类别获取所有叶节点
    
    Args:
        tree: 树的BFS表示
        
    Returns:
        leaf_nodes_by_class: 按类别分组的叶节点索引字典
    """
    leaf_nodes_by_class = defaultdict(list)
    
    for i, node in enumerate(tree):
        if node is not None and node["role"] == "D":  # 移除了 and node["triples"] 条件
            class_key = tuple(sorted(node["triples"]))
            leaf_nodes_by_class[class_key].append(i)
    
    return leaf_nodes_by_class

def generate_sample_for_path(tree, path, features):
    """
    为给定路径生成一个样本
    
    Args:
        tree: 树的BFS表示
        path: 从根节点到叶节点的路径
        features: 所有特征的列表
        
    Returns:
        sample: 生成的样本，特征值为0或1
    """
    # 初始化样本，所有特征都为0
    sample = {feature: 0 for feature in features}
    
    # 设置路径上的条件
    for idx in path[:-1]:  # 排除最后一个节点（叶节点）
        if tree[idx] is not None and tree[idx]["role"] == "C":
            for condition in tree[idx]["triples"]:
                sample[condition] = 1
    
    # 对于dummy特征，赋予随机值
    for feature in features:
        if feature.startswith("dummy_feature_"):
            sample[feature] = random.randint(0, 1)
    
    return sample

def generate_samples(tree, num_samples_per_class=100):
    """
    为决策树中的每个类别生成样本
    
    Args:
        tree: 树的BFS表示
        num_samples_per_class: 每个类别生成的样本数量
        
    Returns:
        samples: 生成的样本列表
        features: 所有特征的列表
        class_indices: 每个样本对应的类别索引
    """
    # 提取所有特征和类别
    features, classes = extract_features_and_classes(tree)
    features = list(features)
    classes = list(classes)
    
    # 按类别获取叶节点
    leaf_nodes_by_class = get_leaf_nodes_by_class(tree)
    
    samples = []
    class_indices = []
    
    # 为每个类别生成等量的样本
    for class_index, class_key in enumerate(classes):
        leaf_nodes = leaf_nodes_by_class[class_key]
        
        # 每个类别生成num_samples_per_class个样本
        samples_for_class = 0
        while samples_for_class < num_samples_per_class:
            # 从该类别的叶节点中随机选择一个
            leaf_index = random.choice(leaf_nodes)
            
            # 找到从根节点到叶节点的路径
            path = find_path_to_leaf(tree, leaf_index)
            
            # 生成样本
            sample = generate_sample_for_path(tree, path, features)
            
            samples.append(sample)
            class_indices.append(class_index)
            
            samples_for_class += 1
    
    return samples, features, class_indices, classes

def save_samples_to_csv(samples, features, class_indices, output_path):
    """
    保存样本到CSV文件
    
    Args:
        samples: 样本列表
        features: 特征列表
        class_indices: 类别索引列表
        output_path: 输出路径
    """
    # 将样本转换为DataFrame
    df_data = []
    for i, sample in enumerate(samples):
        row = [sample[feature] for feature in features]
        row.append(class_indices[i])  # 添加类别标签
        df_data.append(row)
    
    # 创建列名
    column_names = features + ["target"]
    
    # 创建DataFrame
    df = pd.DataFrame(df_data, columns=column_names)
    
    # 保存为CSV
    df.to_csv(output_path, index=False)

def create_class_mapping(classes, output_path):
    """
    创建类别映射文件
    
    Args:
        classes: 类别列表
        output_path: 输出路径
    """
    class_mapping = {i: list(cls) for i, cls in enumerate(classes)}
    with open(output_path, "w") as f:
        json.dump(class_mapping, f, indent=2)

def main():
    # 设置路径和参数
    data_dir = "data/0502_123837/WD"
    generated_subtrees_path = f"{data_dir}/generated_subtrees.json"
    unified_csv_path = f"{data_dir}/unified_synthetic_samples.csv"
    unified_class_mapping_path = f"{data_dir}/unified_class_mapping.json"
    num_samples_per_class = 100
    
    # 创建输出目录（如果不存在）
    os.makedirs(data_dir, exist_ok=True)
    
    # 加载生成的子树
    with open(generated_subtrees_path, "r") as f:
        generated_subtrees = json.load(f)
    
    # 步骤1: 收集所有树的特征和类别
    all_features = set()
    all_classes = set()
    
    # 首先遍历所有树，收集所有特征和类别
    for tree_idx, tree in enumerate(generated_subtrees):
        if tree_idx < 1:  # 跳过索引为0的树（原始未展开的树）
            continue
        
        # 提取当前树的特征和类别
        tree_features, tree_classes = extract_features_and_classes(tree)
        
        # 添加到全局集合
        all_features.update(tree_features)
        all_classes.update(tree_classes)
    
    # 将集合转换为有序的列表，确保特征和类别的顺序与json_tree.py中一致
    # 按字母顺序排序特征
    all_features = sorted(list(all_features))
    
    # 对类别进行排序，确保空类别在最后
    # 首先将每个类别元组转换为字符串，方便排序
    class_strings = []
    for cls in all_classes:
        if not cls:  # 处理空类别
            class_str = "__EMPTY__"  # 与json_tree.py中的处理一致
        else:
            class_str = " ".join(sorted(cls))
        class_strings.append((cls, class_str))
    
    # 按字符串表示排序
    class_strings.sort(key=lambda x: x[1])
    
    # 提取排序后的类别元组
    all_classes = [cls for cls, _ in class_strings]
    
    # 步骤2: 为每个类别收集所有可能的叶节点和对应的树
    # 结构: {class_key: [(tree_idx, leaf_node_idx), ...]}
    class_to_tree_nodes = defaultdict(list)
    
    for tree_idx, tree in enumerate(generated_subtrees):
        if tree_idx < 1:
            continue
            
        # 获取当前树中每个类别的叶节点
        leaf_nodes_by_class = get_leaf_nodes_by_class(tree)
        
        # 添加到全局映射
        for class_key, leaf_nodes in leaf_nodes_by_class.items():
            for leaf_idx in leaf_nodes:
                class_to_tree_nodes[class_key].append((tree_idx, leaf_idx))
    
    # 步骤3: 为每个类别生成样本
    all_samples = []
    all_class_indices = []
    
    for class_idx, class_key in enumerate(all_classes):
        # 获取该类别在所有树中的所有叶节点
        tree_nodes = class_to_tree_nodes[class_key]
        
        if not tree_nodes:  # 如果没有找到该类别的叶节点，则跳过
            continue
            
        # 为该类别生成指定数量的样本
        for _ in range(num_samples_per_class):
            # 随机选择一个树和叶节点
            tree_idx, leaf_idx = random.choice(tree_nodes)
            tree = generated_subtrees[tree_idx]
            
            # 找到从根节点到叶节点的路径
            path = find_path_to_leaf(tree, leaf_idx)
            
            # 生成样本（使用所有特征）
            sample = {feature: 0 for feature in all_features}
            
            # 设置路径上的条件
            for idx in path[:-1]:  # 排除最后一个节点（叶节点）
                if tree[idx] is not None and tree[idx]["role"] == "C":
                    for condition in tree[idx]["triples"]:
                        if condition in sample:  # 确保特征在sample中
                            sample[condition] = 1
            
            # 对于dummy特征，赋予随机值
            for feature in all_features:
                if feature.startswith("dummy_feature_"):
                    sample[feature] = random.randint(0, 1)
            
            all_samples.append(sample)
            all_class_indices.append(class_idx)
    
    # 保存统一的样本集
    save_samples_to_csv(all_samples, all_features, all_class_indices, unified_csv_path)
    
    # 创建统一的类别映射
    create_class_mapping(all_classes, unified_class_mapping_path)
    
    print(f"生成统一数据集:")
    print(f"  样本已保存到 {unified_csv_path}")
    print(f"  类别映射已保存到 {unified_class_mapping_path}")
    print(f"  特征总数: {len(all_features)}")
    print(f"  类别总数: {len(all_classes)}")
    print(f"  样本总数: {len(all_samples)}")
    print(f"  每个类别的样本数: {num_samples_per_class}")
    print(f"  特征顺序: {all_features}")
    print(f"  类别映射: {class_to_string_mapping(all_classes)}")
    
    # 可选: 为单独的树生成样本（如果需要）
    generate_individual_tree_samples = False
    if generate_individual_tree_samples:
        for tree_idx, tree in enumerate(generated_subtrees):
            if tree_idx < 1:
                continue
            
            output_csv_path = f"{data_dir}/synthetic_samples_{tree_idx}.csv"
            class_mapping_path = f"{data_dir}/class_mapping_{tree_idx}.json"
            
            # 生成样本
            samples, features, class_indices, classes = generate_samples(
                tree, 
                num_samples_per_class
            )
            
            # 保存样本到CSV
            save_samples_to_csv(samples, features, class_indices, output_csv_path)
            
            # 创建类别映射文件
            create_class_mapping(classes, class_mapping_path)
            
            print(f"处理树 {tree_idx}:")
            print(f"  生成的样本已保存到 {output_csv_path}")
            print(f"  特征数量: {len(features)}")
            print(f"  类别数量: {len(classes)}")
            print(f"  样本总数: {len(samples)}")
            print(f"  每个类别的样本数量: {num_samples_per_class}")
            print(f"  类别映射已保存到 {class_mapping_path}")
            print()

def class_to_string_mapping(classes):
    """
    将类别元组列表转换为字符串表示的映射，方便调试
    """
    result = {}
    for i, cls in enumerate(classes):
        if not cls:
            result[i] = "__EMPTY__"
        else:
            result[i] = " ".join(sorted(cls))
    return result
            
if __name__ == "__main__":
    main()
