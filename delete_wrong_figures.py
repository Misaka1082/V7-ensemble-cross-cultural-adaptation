"""删除错误的图表文件"""
import os
import glob

# 要删除的图表编号
figures_to_delete = [
    'Figure_4.5_France_Feature_Importance',
    'Figure_4.6_Feature_Importance_Comparison',
    'Figure_4.7_HK_Interaction_Heatmap',
    'Figure_4.8_France_Interaction_Heatmap',
    'Figure_4.9_HK_Cultural_Openness_Interaction',
    'Figure_4.10_France_Maintenance_Support_Interaction',
    'Figure_4.11_Three_Way_Interaction',
    'Figure_4.12_3D_Communication_Model',
    'Figure_4.13_Openness_Inverted_U',
    'Figure_4.14_Autonomy_Cross_Cultural',
    'Figure_4.15_Family_Support_Inverted_U',
    'Figure_4.16_Months_U_Shape',
    'Figure_4.17_Moderation_Principle',
    'Figure_4.18_Radar_Chart',
    'Figure_4.19_SHAP_Waterfall',
    'Figure_4.20_Residual_Diagnostics'
]

output_dir = 'academic_figures'
deleted_count = 0

for fig_name in figures_to_delete:
    # 删除所有格式
    for ext in ['png', 'pdf', 'eps', 'svg']:
        file_path = os.path.join(output_dir, f'{fig_name}.{ext}')
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f'已删除: {file_path}')
            deleted_count += 1

print(f'\n总共删除了 {deleted_count} 个文件')
