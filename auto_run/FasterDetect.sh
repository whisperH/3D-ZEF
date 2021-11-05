#!/bin/bash
# dos2unix filename
# 使用vi打开文本文件
# vi dos.txt
# 命令模式下输入
# :set fileformat=unix
# :w
cd /home/huangjinze/code/3D-ZeF


python threading_detect.py --DayTank D1_T1 --gpuno 0
python threading_detect.py --DayTank D1_T3 --gpuno 1
python threading_detect.py --DayTank D1_T5 --gpuno 2
python threading_detect.py --DayTank D2_T2 --gpuno 3
python threading_detect.py --DayTank D2_T4 --gpuno 4

python threading_detect.py --DayTank D1_T2 --gpuno 0
python threading_detect.py --DayTank D1_T4 --gpuno 1
python threading_detect.py --DayTank D2_T1 --gpuno 2
python threading_detect.py --DayTank D2_T3 --gpuno 3
python threading_detect.py --DayTank D2_T5 --gpuno 4