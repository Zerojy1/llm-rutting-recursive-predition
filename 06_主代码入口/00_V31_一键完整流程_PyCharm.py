from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
STAGES=[
"01_V31_构建母数据库_PyCharm.py",
"02_V31_构建千问时序数据_PyCharm.py",
"03_V31_训练并重拟合千问_PyCharm.py",
"04A_V31_训练TimeGAN与TimeWeaver基线_PyCharm.py",
"04B_V31_生成器质量与可控性审计_PyCharm.py",
"05_V31_TDR过拟合稳定性筛选_PyCharm.py",
"06_V31_等预算真实模拟替代主实验_PyCharm.py",
"07_V31_状态空间与模型容量边界_PyCharm.py",
"08_V31_稀疏监测与OpenLoop汇总_PyCharm.py",
"09_V31_Cluster统计与论文主图_PyCharm.py",
]
if __name__=="__main__":
    for stage in STAGES:
        print("\n>>> RUN",stage,flush=True)
        subprocess.run([sys.executable,str(ROOT/stage)],check=True)
