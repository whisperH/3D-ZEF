import datetime
import os, sys
from multiprocessing import Pool

sys.path.append("../../")
sys.path.append(".")
from common.utility import *


# python中的多线程无法利用多核优势，
# 如果想要充分地使用多核CPU的资源，
# 在python中大部分情况需要使用多进程。
# Python提供了multiprocessing。

def execCmd(cmd):
    try:
        print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
    except:
        print('%s\t 运行失败' % (cmd))


if __name__ == '__main__':
    # 是否需要并行运行
    if_parallel = True
    gpustr = [0, 1, 2, 3, 4, 5, 6, 7]
    exp_floder = os.path.join("/home/data/HJZ/zef/exp_pre")
    config_folder = exp_floder

    # batch_size 个进程，每个进程切 lasting_no * split_time 秒
    split_time = 60
    lasting_no = 10
    batch_size = 5
    raw_video_list = []
    for video_name in os.listdir(exp_floder):
        if video_name.endswith(".mp4"):
            start_time = load_Video_start_time(config_folder, video_name)
            for ibatch in range(batch_size):
                current_time_date = time.strptime(start_time, "%Y_%m_%d_%H_%M_%S")
                end_unix_time = time.mktime(current_time_date) + ibatch * lasting_no * split_time
                new_start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(end_unix_time))
                raw_video_list.append([video_name, new_start_time, lasting_no])

            # 最后一个进程，不确定结果，所以需要剪切剩下的时间段
            current_time_date = time.strptime(start_time, "%Y_%m_%d_%H_%M_%S")
            end_unix_time = time.mktime(current_time_date) + batch_size * lasting_no * split_time
            new_start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(end_unix_time))
            raw_video_list.append([video_name, new_start_time, 100000])

    # 吧 raw_video_list 中的元组按照 start_time 进行排序
    cmds = []
    for ivideo, start_time, lasting_no in sorted(raw_video_list, key=lambda cmd_info: cmd_info[1]):
        cmd_str = f"python modules/dataset_processing/CutTankROI.py " \
            f"--exp_floder {exp_floder} " \
            f"--video_name {ivideo} " \
            f"--start_time {start_time} " \
            f"--lasting_no {lasting_no} " \
            f"--split_time {split_time} "
        cmds.append(cmd_str)

    print("*****************************************************")
    print(f"{len(cmds)} are need to be processed")
    print("*****************************************************")
    if if_parallel:
        # 并行
        pool = Pool(8)
        pool.map(execCmd, cmds)
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()
    else:
        # 串行
        for cmd in cmds:
            try:
                print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
                os.system(cmd)
                print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
            except:
                print('%s\t 运行失败' % (cmd))
