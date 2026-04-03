"""清理无用文件脚本"""
import os

# 要删除的文件列表
files_to_delete = [
    r"F:\Project\4_1_9_final\FranchN=249（1）.xlsx",
    r"F:\Project\4_1_9_final\跨文化适应V7模型完整研究报告_香港与法国对比 - 副本.md",
    r"F:\Project\4_1_9_final\任务完成情况分析.md",
    r"F:\Project\4_1_9_final\organize_files.py",
    r"F:\Project\4_1_9_final\generate_chapter5_6_revision_guide.py",
]

# 要删除的目录列表
dirs_to_delete = [
    r"F:\Project\4_1_9_final\__pycache__",
    r"F:\Project\4_1_9_final\catboost_info",
]

print("开始清理文件...")

# 删除文件
for file_path in files_to_delete:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"✓ 已删除: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"✗ 删除失败: {os.path.basename(file_path)} - {e}")
    else:
        print(f"- 文件不存在: {os.path.basename(file_path)}")

# 删除目录
import shutil
for dir_path in dirs_to_delete:
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"✓ 已删除目录: {os.path.basename(dir_path)}")
        except Exception as e:
            print(f"✗ 删除目录失败: {os.path.basename(dir_path)} - {e}")
    else:
        print(f"- 目录不存在: {os.path.basename(dir_path)}")

print("\n清理完成！")
