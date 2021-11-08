import paramiko
import sys

def deleteRemoteFile(dt, ip, name, passwd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())#第一次登录的认证信息
    ssh.connect(hostname=ip, port=22, username=name, password=passwd) # 连接服务器
    stdin, stdout, stderr = ssh.exec_command(f'rm -rf {dt}') # 执行命令
    ssh.close()


def uploadFile2Remote(win_file, linux_file, ip, name, passwd):
    transport = paramiko.Transport((ip, 22))
    transport.connect(username=name, password=passwd)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put(win_file, linux_file)
    transport.close()

if __name__ == '__main__':
    server_info = [
        ['10.2.151.127', 'huangjinze', 'huangjinze33'],
        ['10.2.151.128', 'huangjinze', 'huangjinze33'],
        ['10.2.151.129', 'huangjinze', 'huangjinze33'],
    ]
    for ip, name, passwd in server_info:
        files = [
            ['e:\code/3D-ZeF/common/utility.py', '/home/huangjinze/code/3D-ZeF/common/utility.py'],
            ['e:\code/3D-ZeF/modules/detection/BgDetector.py', '/home/huangjinze/code/3D-ZeF/modules/detection/BgDetector.py'],
            ['e:\code/3D-ZeF/threading_bgdetect.py', '/home/huangjinze/code/3D-ZeF/threading_bgdetect.py'],
        ]
        for win_file, linux_file in files:
            deleteRemoteFile(win_file, ip, name, passwd)
            uploadFile2Remote(win_file, linux_file, ip, name, passwd)