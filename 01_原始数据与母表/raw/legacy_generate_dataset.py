import pandas as pd
import numpy as np
import sys

def generate_perfect_dataset():
    print("开始构建 18 维具备物理语义的结构特征正交矩阵...")

    # 1. 初始化结构特征矩阵 (使用工程语义变量名替代 x3-x20)
    str_names = [f'STR{i}' for i in range(1, 20)]

    # 按照面层(Surf)、中面层(Mid)、下面层(Bot)、基层(Base)分类命名
    semantic_cols = [
        'Surf_AC13_I_SBS', 'Surf_AC13_II_SBS', 'Surf_SMA13', 'Surf_PAC13',
        'Mid_AC20_AH30', 'Mid_AC20_SBS', 'Mid_AC20_AH50',
        'Bot_AC25_AH30', 'Bot_AC25_AH50', 'Bot_AC25_AH70', 'Bot_AC25_Re_AH70', 'Bot_AC10_SBS',
        'Base_CBG_A', 'Base_CBG_B', 'Base_CS', 'Base_CC', 'Base_LCC', 'Base_GA'
    ]

    df_struct = pd.DataFrame(0, index=str_names, columns=semantic_cols)

    # ==========================================
    # 2. 注入物理厚度映射规则 (严格遵循指定的标准映射)
    # ==========================================
    # 🟢 第 I 类（半刚性基层结构）
    df_struct.loc['STR1', ['Surf_AC13_I_SBS', 'Mid_AC20_AH30', 'Base_CBG_A', 'Base_CS']] = [4, 8, 40, 40]
    df_struct.loc['STR2', ['Surf_AC13_I_SBS', 'Mid_AC20_AH30', 'Base_CBG_A', 'Base_CS']] = [4, 8, 40, 20]
    df_struct.loc['STR3', ['Surf_AC13_I_SBS', 'Mid_AC20_AH30', 'Base_CBG_A', 'Base_GA']] = [4, 8, 40, 20]

    # 🟢 第 II 类（刚性复合式基层结构）
    df_struct.loc['STR4', ['Surf_AC13_II_SBS', 'Mid_AC20_AH30', 'Bot_AC10_SBS', 'Base_LCC', 'Base_CBG_A', 'Base_CS']] = [4, 6, 2, 24, 20, 20]
    df_struct.loc['STR5', ['Surf_AC13_II_SBS', 'Mid_AC20_SBS', 'Bot_AC10_SBS', 'Base_CC', 'Base_CBG_A', 'Base_CS']] = [4, 6, 2, 24, 20, 20]

    # 🟢 第 III 类（半刚性基层结构）
    df_struct.loc['STR6', ['Surf_AC13_II_SBS', 'Bot_AC25_AH30', 'Bot_AC10_SBS', 'Base_CBG_A', 'Base_CS']] = [4, 10, 2, 38, 20]
    df_struct.loc['STR7', ['Surf_AC13_II_SBS', 'Mid_AC20_SBS', 'Bot_AC25_AH70', 'Base_CBG_A', 'Base_CS']] = [4, 6, 8, 38, 20]
    df_struct.loc['STR8', ['Surf_AC13_II_SBS', 'Mid_AC20_SBS', 'Bot_AC25_AH70', 'Base_CBG_B', 'Base_CS']] = [4, 6, 8, 38, 20]
    df_struct.loc['STR9', ['Surf_PAC13', 'Mid_AC20_SBS', 'Bot_AC25_AH70', 'Base_CBG_B', 'Base_CS']] = [4, 6, 8, 38, 20]

    # 🟢 第 IV 类（倒装式基层结构）
    df_struct.loc['STR10', ['Surf_AC13_I_SBS', 'Mid_AC20_SBS', 'Bot_AC25_AH70', 'Bot_AC10_SBS', 'Base_GA', 'Base_CBG_B', 'Base_CS']] = [4, 6, 16, 2, 20, 20, 20]
    df_struct.loc['STR12', ['Surf_AC13_I_SBS', 'Mid_AC20_SBS', 'Bot_AC25_AH70', 'Base_GA', 'Base_CBG_B', 'Base_CS']] = [4, 8, 12, 20, 20, 20]

    # 🟢 第 V 类（厚沥青混凝土结构 I）
    df_struct.loc['STR11', ['Surf_AC13_I_SBS', 'Mid_AC20_SBS', 'Bot_AC25_AH70', 'Base_CBG_A', 'Base_CS']] = [4, 6, 18, 40, 20]
    df_struct.loc['STR13', ['Surf_AC13_I_SBS', 'Mid_AC20_SBS', 'Bot_AC25_AH70', 'Base_CBG_A', 'Base_CS']] = [4, 8, 12, 40, 20]
    df_struct.loc['STR14', ['Surf_AC13_I_SBS', 'Mid_AC20_SBS', 'Bot_AC25_Re_AH70', 'Base_CBG_B', 'Base_CS']] = [4, 8, 12, 40, 20]

    # 🟢 第 VI 类（厚沥青混凝土结构 II）
    df_struct.loc['STR15', ['Surf_AC13_I_SBS', 'Mid_AC20_AH50', 'Bot_AC25_AH50', 'Base_CBG_A', 'Base_GA']] = [4, 8, 24, 20, 44]
    df_struct.loc['STR16', ['Surf_SMA13', 'Mid_AC20_SBS', 'Bot_AC25_AH70', 'Base_CBG_A', 'Base_CS']] = [4, 8, 24, 20, 20]
    df_struct.loc['STR17', ['Surf_SMA13', 'Mid_AC20_AH30', 'Bot_AC25_AH30', 'Base_CBG_A', 'Base_CS']] = [4, 8, 24, 20, 20]

    # 🟢 第 VII 类（全厚式结构）
    df_struct.loc['STR18', ['Surf_SMA13', 'Mid_AC20_AH50', 'Bot_AC25_AH50', 'Bot_AC10_SBS', 'Base_GA']] = [4, 8, 36, 4, 48]
    df_struct.loc['STR19', ['Surf_SMA13', 'Mid_AC20_AH50', 'Bot_AC25_AH30', 'Base_CBG_B']] = [4, 8, 36, 20]

    df_struct['STR_name'] = df_struct.index
    print("结构特征矩阵构建完成！")

    # ==========================================
    # 3. 读取 Excel 原始宽表并进行无损面板重塑
    # ==========================================
    print("正在读取并重塑原始 Excel 宽表数据...")
    # 注意：请根据你的实际路径修改以下 raw_file 和 output_name
    raw_file = r'C:\Users\zianyy\Desktop\论文修改\数据集处理\足尺环道车辙数据.xlsx'

    try:
        raw_df = pd.read_excel(raw_file, header=1)
    except Exception as e:
        print(f"\n【错误】无法读取文件：{e}")
        sys.exit(1)

    # 赋予 x1, x2 物理意义名称
    raw_df = raw_df.rename(columns={
        'Unnamed: 3': 'Cum_Load_10k',  # 当量设计轴载累计作用次数(万次)
        'Unnamed: 4': 'Avg_Temp_C'  # 平均气温(℃)
    })

    # 宽表转长表
    str_cols = [f'STR{i}' for i in range(1, 20)]
    melted_df = raw_df[['Cum_Load_10k', 'Avg_Temp_C'] + str_cols].melt(
        id_vars=['Cum_Load_10k', 'Avg_Temp_C'],
        value_vars=str_cols,
        var_name='STR_name',
        value_name='Rutting_Depth_0_1mm'  # y变量也语义化
    )

    # ==========================================
    # 4. 原子级无损去空与排序
    # ==========================================
    melted_df['orig_index'] = melted_df.groupby('STR_name').cumcount()
    melted_df['STR_num'] = melted_df['STR_name'].str.replace('STR', '').astype(int)

    clean_df = melted_df.dropna(subset=['Rutting_Depth_0_1mm']).copy()
    clean_df = clean_df.sort_values(['orig_index', 'STR_num'])

    # ==========================================
    # 5. 特征融合与导出
    # ==========================================
    print("正在融合结构特征...")
    final_df = pd.merge(clean_df, df_struct, on='STR_name', how='left')

    # 整理最终保留的列：时间/荷载/温度 + 18维物理厚度 + 目标变量(车辙)
    final_columns = ['Cum_Load_10k', 'Avg_Temp_C'] + semantic_cols + ['Rutting_Depth_0_1mm']
    final_df = final_df[final_columns]

    output_name = r'C:\Users\zianyy\Desktop\论文修改\数据集处理\ruttingData_Final_Corrected.xlsx'

    try:
        final_df.to_excel(output_name, index=False)
    except Exception as e:
        print(f"\n【错误】导出失败：{e}")
        sys.exit(1)

    print("\n========== 数据处理报告 ==========")
    print(f"数据总行数: {final_df.shape[0]}")
    print(f"特征总列数: {final_df.shape[1]} (含荷载、环境、18种材料层、车辙深度)")
    print(f"缺失值总数: {final_df.isna().sum().sum()}")
    print(f"输出文件已保存至: {output_name}")
    print("==================================\n")

if __name__ == "__main__":
    generate_perfect_dataset()