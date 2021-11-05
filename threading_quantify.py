import datetime
import os
from multiprocessing import Pool
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
    indicator_floder = os.path.join("/home/data/HJZ/zef/exp_prebak/indicators")
    tracker_floder = os.path.join("/home/data/HJZ/zef/exp_prebak/sort_tracker")
    if not os.path.isdir(indicator_floder):
        os.mkdir(indicator_floder)

    processed_list = []
    for files in os.listdir(indicator_floder):
        if files.endswith(".csv"):
            NO, region_name = files.split("_")
            processed_list.append(NO)

    track_files_tops = sorted(os.listdir(os.path.join(tracker_floder, "D01")))
    track_file_rights = sorted(os.listdir(os.path.join(tracker_floder, "D02")))
    track_file_lefts = sorted(os.listdir(os.path.join(tracker_floder, "D04")))

    end_no = min(len(track_files_tops), len(track_file_lefts), len(track_file_rights))
    batch_size = 1
    batch_nos = [(i*batch_size, (i+1)*batch_size) for i in range(end_no // batch_size)]

    # generate batch

    # 需要执行的命令列表
    cmds = []
    for ino in batch_nos:

        if [f"{str(batch_nos[0])}-{str(batch_nos[1])}"] in processed_list:
            print(f"{files}: {str(batch_nos[0])}-{str(batch_nos[1])} has been processed")
            continue
        else:
            cmd_str = f"python modules/quantify/basicIndex.py " \
                f"-o /home/data/HJZ/zef/exp_prebak/indicators " \
                f"-cp /home/data/HJZ/zef/exp_prebak/ " \
                f"-tr /home/data/HJZ/zef/exp_prebak/sort_tracker/ " \
                f"-if NO_RegionName " \
                f"-fn 1 -no {str(ino[0])}-{str(ino[1])}"
            cmds.append(cmd_str)

    print("*****************************************************")
    print(f"{len(cmds)} are need to be processed")
    print("*****************************************************")
    if if_parallel:
        # 并行
        pool = Pool(16)
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


