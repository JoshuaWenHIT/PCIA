"""
External Version 1.4
Internal Version 5.4
2021/04/30 by Joshua Wen & BWH & WZH from HEU & XiaoLong
"""
import sys
import torch.multiprocessing

from lib.rpc import RpcClient
from lib.live import streaming
from lib.live_plug import streaming_plug


class Logger(object):
    def __init__(self, logFile="./log/Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Run(object):
    def __init__(self, parameters, opt):
        import torch.multiprocessing
        torch.multiprocessing.set_start_method('spawn')
        self.share_var = torch.multiprocessing.Manager().Value(int, 0)
        self.share_var_1 = torch.multiprocessing.Manager().list()
        self.share_var_2 = torch.multiprocessing.Manager().list()
        self.share_var_3 = torch.multiprocessing.Manager().Value(int, 0)
        self.share_var_4 = torch.multiprocessing.Manager().list()
        self.share_var_5 = torch.multiprocessing.Manager().Queue()

        self.share_lock = torch.multiprocessing.Manager().Lock()
        self.share_lock_1 = torch.multiprocessing.Manager().Lock()
        self.share_lock_2 = torch.multiprocessing.Manager().Lock()
        self.share_lock_3 = torch.multiprocessing.Manager().Lock()
        self.share_lock_4 = torch.multiprocessing.Manager().Lock()

        self.parameters = parameters
        self.opt = opt

    def start_MQ(self, share_var, share_var_1, share_var_2, share_var_3, share_var_4, share_lock, share_lock_1, share_lock_2,
                 share_lock_3, share_lock_4, parameters):
        mqtt = RpcClient(share_var, share_var_1, share_var_2, share_var_3, share_var_4, share_lock, share_lock_1,
                         share_lock_2, share_lock_3, share_lock_4, parameters)
        mqtt.run()

    def get_process_list(self):
        process_list = []

        # mq_process = torch.multiprocessing.Process(target=self.start_MQ,
        #                                            args=(self.share_var, self.share_var_1, self.share_var_2,
        #                                                  self.share_var_3, self.share_var_4, self.share_lock,
        #                                                  self.share_lock_1, self.share_lock_2, self.share_lock_3,
        #                                                  self.share_lock_4, self.parameters))
        # process_list.append(mq_process)

        stream_process = torch.multiprocessing.Process(target=streaming,
                                                       args=(self.share_var, self.share_var_1, self.share_var_4,
                                                             self.share_var_5, self.share_lock, self.share_lock_1,
                                                             self.share_lock_4, self.parameters, self.opt))
        process_list.append(stream_process)
        if self.parameters.live:
            live_process = torch.multiprocessing.Process(target=streaming_plug,
                                                         args=(self.share_var, self.share_lock, self.share_var_5,
                                                               self.parameters))
            process_list.append(live_process)
        return process_list

    def run(self):
        process_list = self.get_process_list()
        for process in process_list:
            process.start()
        for process in process_list:
            process.join()
