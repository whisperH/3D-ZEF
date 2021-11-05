import datetime
import os
from multiprocessing import Pool
from common.utility import load_EXP_region_name
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
    tracker_name = 'sort_tracker'
    if_parallel = True
    exp_floder = os.path.join(f"E:\\data\\OCU_ZeF\\")
    # detect_floder = os.path.join("/home/data/HJZ/zef/processed")
    detect_floder = os.path.join("E:\\data\\OCU_ZeF\\processed")
    # tracker_floder = os.path.join("/home/data/HJZ/zef/exp_pre/tracker")
    tracker_floder = os.path.join(f"E:\\data\\OCU_ZeF\\{tracker_name}")

    config_folder = exp_floder
    region_names = load_EXP_region_name(config_folder)
    processed_list = []
    detect_files = []

    for region_name in region_names:
        processed_file = os.path.join(tracker_floder, region_name)
        if not os.path.isdir(processed_file):
            os.makedirs(processed_file)
        processed_list.extend(os.listdir(processed_file))


        for file in os.listdir(os.path.join(detect_floder, region_name)):
            detect_file = os.path.join(detect_floder, region_name, file)
            detect_files.append(detect_file)

    # 需要执行的命令列表
    cmds = []
    for idetfile in sorted(detect_files):
        # itrackfile = idetfile.replace("processed", tracker_name)
        detect_path, filename = os.path.split(idetfile)
        trackpath = detect_path.replace("processed", tracker_name)

        trackfile = os.path.join(trackpath, filename)
        # print(trackfile)
        # itrackfile = os.path.join(track_path, track_filename)
        if os.path.split(trackfile)[-1] in processed_list:
            print(f"{trackfile} has been processed")
            continue
        else:
            cmd_str = f"python modules/tracking/SortTracker.py " \
                f"--detection_file {idetfile} --track_file {trackfile} --startFrame 0 " \
                f"-tfloder {trackpath} "
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


