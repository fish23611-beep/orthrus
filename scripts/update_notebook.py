#!/usr/bin/env python3
"""
Update the All-in-One Colab Notebook with bounded-memory preprocessing features.

This script adds:
1. Resource pre-check section (RAM, GPU, disk)
2. Preprocess substage parameter support
3. Enhanced completion checking with markers
4. Memory tracking information
"""
import json
import sys


def load_notebook(path):
    with open(path) as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)


def find_cell_index(nb, marker_text):
    """Find the index of a cell containing marker_text in its source."""
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))
        if marker_text in source:
            return i
    return -1


def find_next_code_cell_index(nb, start_idx):
    """Find the next code cell after start_idx."""
    for i in range(start_idx + 1, len(nb['cells'])):
        if nb['cells'][i].get('cell_type') == 'code':
            return i
    return -1


def create_markdown_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content
    }


def create_code_cell(content):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content
    }


def update_notebook(nb):
    """Update the notebook with bounded-memory preprocessing features."""
    
    # =========================================================================
    # 1. Update header with new version info
    # =========================================================================
    header_cell = nb['cells'][0]
    header_source = ''.join(header_cell['source'])
    header_source = header_source.replace(
        '**Version**: 1.0.0',
        '**Version**: 1.1.0'
    )
    header_source = header_source.replace(
        '**Commit**: `0a7ab00bb0900df5afd4ecb01359ab10ad37c60e`',
        '**Commit**: `fix/c8-preprocess-memory` (local development)'
    )
    header_cell['source'] = [header_source]
    
    # =========================================================================
    # 2. Add resource pre-check section after GPU/CUDA section (cell 7-8)
    # =========================================================================
    gpu_cuda_idx = find_cell_index(nb, '## 3. GPU / CUDA 验证')
    
    if gpu_cuda_idx != -1:
        # Find the end of the GPU section (next markdown header)
        end_idx = gpu_cuda_idx + 1
        while end_idx < len(nb['cells']):
            cell = nb['cells'][end_idx]
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                if source.startswith('## '):
                    break
            end_idx += 1
        
        # Insert resource pre-check cell before next section
        resource_check_cell = create_code_cell('''# ============================================================================
# 资源预检：显示系统资源状态
# ============================================================================
import psutil
import shutil

print("=" * 60)
print("资源预检")
print("=" * 60)

# CPU RAM
ram = psutil.virtual_memory()
print(f"总 RAM:     {ram.total / (1024**3):.1f} GB")
print(f"可用 RAM:   {ram.available / (1024**3):.1f} GB")
print(f"RAM 使用率: {ram.percent}%")
if ram.percent > 80:
    print("⚠️  警告: RAM 使用率较高，可能影响 preprocessing")

# GPU
if shutil.which("nvidia-smi"):
    print("\\nGPU 信息:")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], check=False)
else:
    print("\\nGPU: 不可用 (CPU-only 环境)")

# Disk space
for path in ["/content", str(DRIVE_ROOT)]:
    if os.path.exists(path):
        usage = shutil.disk_usage(path)
        print(f"\\n{path} 磁盘: 总计 {usage.total / (1024**3):.1f} GB, "
              f"可用 {usage.free / (1024**3):.1f} GB ({usage.percent}% used)")

print("=" * 60)
print("资源预检完成")
print("=" * 60)
''')
        
        nb['cells'].insert(end_idx, resource_check_cell)
        print(f"Added resource pre-check cell at index {end_idx}")
    
    # =========================================================================
    # 3. Update preprocess substage parameter section
    # =========================================================================
    # Find the parameter section (cell 2)
    param_cell_idx = find_cell_index(nb, 'DATASET = "THEIA_E3"')
    
    if param_cell_idx != -1:
        param_cell = nb['cells'][param_cell_idx]
        source = ''.join(param_cell['source'])
        
        # Add preprocess substages parameter after FORCE_PREPROCESS
        if 'PREPROCESS_SUBSTAGES' not in source:
            # Find the line with FORCE_PREPROCESS and add after it
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if 'FORCE_PREPROCESS = False' in line:
                    new_lines.append('')
                    new_lines.append('# --- Preprocessing Substages (C8.1 bounded-memory) ---')
                    new_lines.append('# 可选值: "build_graphs" | "embed_nodes" | "embed_edges"')
                    new_lines.append('# 默认运行所有三个阶段，如需单独运行可设置为单个值')
                    new_lines.append('PREPROCESS_SUBSTAGES = "build_graphs,embed_nodes,embed_edges"')
                    new_lines.append('# 示例 - 单独运行各阶段:')
                    new_lines.append('# PREPROCESS_SUBSTAGES = "build_graphs"  # 先运行图构建')
                    new_lines.append('# PREPROCESS_SUBSTAGES = "embed_nodes"  # 然后词向量')
                    new_lines.append('# PREPROCESS_SUBSTAGES = "embed_edges"  # 最后边嵌入')
            
            param_cell['source'] = ['\n'.join(new_lines)]
            print(f"Updated parameter cell at index {param_cell_idx}")
    
    # =========================================================================
    # 4. Update preprocess execution cell to use substages
    # =========================================================================
    preprocess_cell_idx = find_cell_index(nb, 'RUN_PREPROCESS=False')
    
    if preprocess_cell_idx != -1:
        preprocess_cell = nb['cells'][preprocess_cell_idx]
        source = ''.join(preprocess_cell['source'])
        
        # Update the command to include --preprocess-substages
        if '--preprocess-substages' not in source:
            # Update command line
            source = source.replace(
                '"--stages", "preprocess", "--skip-tracing"',
                '"--stages", "preprocess", "--preprocess-substages", PREPROCESS_SUBSTAGES, "--skip-tracing"'
            )
            
            # Add pre-check for completion markers
            if 'completion marker' not in source:
                # Add marker checking logic
                marker_check = '''
# C8.1: 检查 completion markers 以支持增量运行
graphs_dir = ARTIFACT_ROOT / "graphs"
build_marker = graphs_dir / ".preprocess_build_graphs_complete"
embed_nodes_marker = ARTIFACT_ROOT / "word2vec" / ".preprocess_embed_nodes_complete"
embed_edges_marker = ARTIFACT_ROOT / "edge_embeds" / ".preprocess_embed_edges_complete"

# 显示当前状态
print(f"build_graphs marker: {'✓' if build_marker.exists() else '✗'}")
print(f"embed_nodes marker: {'✓' if embed_nodes_marker.exists() else '✗'}")
print(f"embed_edges marker: {'✓' if embed_edges_marker.exists() else '✗'}")

# 如果运行的是 build_graphs 但已有完整产物，跳过其他阶段
if PREPROCESS_SUBSTAGES == "build_graphs,embed_nodes,embed_edges":
    if build_marker.exists() and embed_nodes_marker.exists() and embed_edges_marker.exists():
        print("检测到完整预处理产物，跳过预处理。")
        print("如需强制重新运行，请设置 FORCE_PREPROCESS = True")
        FORCE_PREPROCESS = False
'''
                # Find a good insertion point (before the command)
                if 'command = [' in source:
                    # Insert marker check before the command
                    parts = source.split('command = [')
                    source = parts[0] + marker_check + '\ncommand = [' + parts[1]
            
            preprocess_cell['source'] = [source]
            print(f"Updated preprocess cell at index {preprocess_cell_idx}")
    
    # =========================================================================
    # 5. Update artifacts check to use markers
    # =========================================================================
    artifacts_cell_idx = find_cell_index(nb, 'required_paths = {')
    
    if artifacts_cell_idx != -1:
        artifacts_cell = nb['cells'][artifacts_cell_idx]
        source = ''.join(artifacts_cell['source'])
        
        # Add marker-based checking
        if '.preprocess_' not in source:
            marker_check_addition = '''
# C8.1: 使用 completion markers 进行更精确的状态检测
def check_completion_marker(artifact_dir, marker_name):
    marker_path = artifact_dir / marker_name
    return marker_path.exists()

# 检查 markers
graphs_dir = Path(cfg.graph_construction.build_graphs._graphs_dir)
word2vec_dir = Path(cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir)
edge_embeds_dir = Path(cfg.edge_featurization.embed_edges._edge_embeds_dir)

marker_status = {
    "build_graphs": check_completion_marker(graphs_dir, ".preprocess_build_graphs_complete"),
    "embed_nodes": check_completion_marker(word2vec_dir, ".preprocess_embed_nodes_complete"),
    "embed_edges": check_completion_marker(edge_embeds_dir, ".preprocess_embed_edges_complete"),
}
print("\\nC8.1 Completion Markers:")
for stage, complete in marker_status.items():
    print(f"  {stage}: {'✓' if complete else '✗'}")
'''
            # Insert after the required_paths definition
            if 'def visible_entries' in source:
                parts = source.split('def visible_entries', 1)
                source = parts[0] + marker_check_addition + '\n\ndef visible_entries' + parts[1]
            
            artifacts_cell['source'] = [source]
            print(f"Updated artifacts cell at index {artifacts_cell_idx}")
    
    return nb


def main():
    notebook_path = 'notebooks/ORTHRUS_MSTC_PIDS_AllInOne_Colab.ipynb'
    
    print(f"Loading notebook: {notebook_path}")
    nb = load_notebook(notebook_path)
    
    print("Updating notebook with C8.1 bounded-memory preprocessing features...")
    nb = update_notebook(nb)
    
    print(f"Saving notebook: {notebook_path}")
    save_notebook(nb, notebook_path)
    
    print("Done!")


if __name__ == '__main__':
    main()
