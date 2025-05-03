import json
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict

def predict_with_tree(tree, sample):
    """
    使用决策树对单个样本进行预测
    
    Args:
        tree: BFS格式的树
        sample: 单个样本的特征字典
    
    Returns:
        预测的类别
    """
    # 从根节点开始
    current_node_idx = 0
    
    while True:
        current_node = tree[current_node_idx]
        
        # 如果是叶节点，返回其类别
        if current_node is None or current_node["role"] == "D":
            return tuple(sorted(current_node["triples"] if current_node else []))
        
        # 如果是内部节点，根据条件决定走左子树还是右子树
        if current_node["role"] == "C":
            # 获取条件
            conditions = current_node["triples"]
            
            # 检查条件是否满足
            condition_satisfied = False
            for condition in conditions:
                if condition in sample and sample[condition] == 1:
                    condition_satisfied = True
                    break
            
            # 根据条件满足情况决定走左子树还是右子树
            if condition_satisfied:
                # 左子树
                current_node_idx = 2 * current_node_idx + 1
            else:
                # 右子树
                current_node_idx = 2 * current_node_idx + 2
        
        # 如果超出树的范围，返回空列表
        if current_node_idx >= len(tree) or tree[current_node_idx] is None:
            return tuple([])

def count_internal_nodes(tree):
    """
    计算树的内部节点数量
    
    Args:
        tree: BFS格式的树
    
    Returns:
        内部节点数量
    """
    count = 0
    for node in tree:
        if node is not None and node["role"] == "C":
            count += 1
    return count

def evaluate_trees(trees, data_csv, class_mapping_json):
    """
    评估多棵树的性能
    
    Args:
        trees: BFS格式的树列表
        data_csv: 样本数据CSV文件路径
        class_mapping_json: 类别映射JSON文件路径
    
    Returns:
        每棵树的评估结果字典
    """
    # 加载样本数据
    df = pd.read_csv(data_csv)
    
    # 分离特征和目标
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 加载类别映射
    with open(class_mapping_json, 'r') as f:
        class_mapping = json.load(f)
    
    # 反向映射，从类别元组到索引
    class_to_idx = {}
    for idx, class_list in class_mapping.items():
        class_to_idx[tuple(sorted(class_list))] = int(idx)
    
    # 评估每棵树
    results = {}
    
    for tree_idx, tree in enumerate(trees):
        if tree_idx < 1:  # 跳过索引为0的树
            continue
        
        # 将样本转换为字典列表，方便预测
        samples = []
        for _, row in X.iterrows():
            sample = row.to_dict()
            samples.append(sample)
        
        # 对每个样本进行预测
        predicted_classes = []
        for sample in samples:
            predicted_class = predict_with_tree(tree, sample)
            predicted_idx = class_to_idx.get(predicted_class, -1)  # 如果找不到映射，返回-1
            predicted_classes.append(predicted_idx)
        
        # 评估指标
        accuracy = accuracy_score(y, predicted_classes)
        f1_macro = f1_score(y, predicted_classes, average='macro')
        complexity = count_internal_nodes(tree)
        
        # 保存结果
        results[tree_idx] = {
            'accuracy': accuracy,
            'f1_score': f1_macro,
            'complexity': complexity
        }
        
        print(f"树 {tree_idx} 的评估结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1 分数: {f1_macro:.4f}")
        print(f"  复杂度 (内部节点数): {complexity}")
        print(f"  分类报告:")
        print(classification_report(y, predicted_classes))
        print()
    
    return results

def main():
    # 设置路径
    data_dir = "data/0502_123837/WD"
    subtrees_path = f"{data_dir}/generated_subtrees.json"
    samples_path = f"{data_dir}/unified_synthetic_samples.csv"
    class_mapping_path = f"{data_dir}/unified_class_mapping.json"
    
    # 加载子树
    with open(subtrees_path, 'r') as f:
        subtrees = json.load(f)
    
    # 评估树
    results = evaluate_trees(subtrees, samples_path, class_mapping_path)
    
    # 保存结果
    output_path = f"{data_dir}/tree_evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"评估结果已保存到 {output_path}")
    
    # 找出最佳树
    best_tree_idx = max(results.keys(), key=lambda k: results[k]['f1_score'])
    print(f"最佳树 (按F1分数): 树 {best_tree_idx}, F1分数 = {results[best_tree_idx]['f1_score']:.4f}")
    
    # 计算不同复杂度树的平均性能
    complexity_to_performance = defaultdict(list)
    for tree_idx, metrics in results.items():
        complexity = metrics['complexity']
        complexity_to_performance[complexity].append(metrics['f1_score'])
    
    print("\n不同复杂度树的平均F1分数:")
    for complexity, scores in sorted(complexity_to_performance.items()):
        avg_score = sum(scores) / len(scores)
        print(f"  复杂度 {complexity}: 平均F1分数 = {avg_score:.4f} (基于 {len(scores)} 棵树)")

if __name__ == "__main__":
    main() 