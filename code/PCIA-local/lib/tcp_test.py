"""
        @author     zmc & Joshua
        @date       2021/06/09
"""


import socket
import threading
import json
from lib.tools import get_gpu_status


class tcp_server:

    def __init__(self, share_var, share_var_1, share_var_4, share_var_6,  share_var_8, share_lock, share_lock_1, share_lock_4, share_lock_6, share_lock_8):

        self.share_var = share_var
        self.share_var_1 = share_var_1
        self.share_var_4 = share_var_4
        self.share_var_6 = share_var_6
        self.share_var_8 = share_var_8

        self.share_lock = share_lock
        self.share_lock_1 = share_lock_1
        self.share_lock_4 = share_lock_4
        self.share_lock_6 = share_lock_6
        self.share_lock_8 = share_lock_8

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("0.0.0.0", 2020))
        self.socket.listen(10)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("Waiting for connection...")

        self.client_thread = []

        self.accept = threading.Thread(target=self.thread_accept, args=())
        self.accept.start()

    def thread_accept(self):
        """
         等待客户端连接线程
        :return:
        """
        while True:
            client_socket, client_addr = self.socket.accept()
            t = threading.Thread(target=self.thread_client_data, args=(client_socket, client_addr))
            t.start()
            self.client_thread.append(t)

    def thread_client_data(self, client_socket, client_addr):
        """
         处理客户端数据线程
        :return:
        """
        print("客户端", client_addr, "已连接.")

        while True:

            data = client_socket.recv(1024).decode()
            if not data:
                break

            print("接收到", client_addr, "数据", data)

            response_text = self.todo_recv_order(data)
            print("response_text:", response_text)

            client_socket.send(response_text.encode())
            print()
            print()

        print("客户端", client_addr, "下线了.")

    def is_json(self, myjson):
        try:
            json_object = json.loads(myjson)
        except ValueError:
            return False
        return True

    def todo_recv_order(self, _str):
        blog_ck1 = None
        blog_ck2 = None
        resultNumber = ""
        inputVideoPath = ""
        outputVideoPath = ""
        orderList = ["setParam", "start", "stop", "getStatus"]

        if self.is_json(_str):
            order = json.loads(_str)
            print(type(order))
            # try:
            order_str = ''
            if "order" in _str and "orderid" in _str and "data" in _str:
                order_str = order['order']
                print(order_str)
                orderid_str = order['orderid']
                print(orderid_str)
                data_str = order['data']
                print(data_str)

                if order_str in orderList:
                    if order_str == 'setParam':
                        self.share_lock_6.acquire()
                        self.share_var_6["cameraIP"] = data_str["cameraIP"]
                        self.share_var_6["rtmpIP"] = data_str["rtmpIP"]
                        self.share_var_6["direction"] = data_str["direction"]
                        self.share_var_6["autoSave"] = data_str["autoSave"]
                        self.share_var_6["live"] = data_str["live"]
                        self.share_lock_6.release()

                        blog = 2
                        self.share_lock.acquire()
                        self.share_var.set(blog)
                        self.share_lock.release()

                        while True:
                            self.share_lock_8.acquire()
                            if "blog_ck1" in self.share_var_8.keys() and "blog_ck2" in self.share_var_8.keys():
                                blog_ck1 = self.share_var_8["blog_ck1"]
                                blog_ck2 = self.share_var_8["blog_ck2"]
                            self.share_lock_8.release()
                            if blog_ck1 == blog_ck2 == "done":
                                break
                        blog = 3
                        self.share_lock.acquire()
                        self.share_var.set(blog)
                        self.share_lock.release()
                        return '{"orderid": "' + orderid_str + '","code": "200","msg": "setParam ok","data": {}}'
                    elif order_str == 'start':
                        self.share_lock_6.acquire()
                        self.share_var_6["uuid"] = data_str["uuid"]
                        self.share_lock_6.release()
                        blog = 1
                        self.share_lock.acquire()
                        self.share_var.set(blog)
                        self.share_lock.release()
                        return '{"orderid": "' + orderid_str + '","code": "200","msg": "start ok","data": {}}'
                    elif order_str == 'stop':
                        blog = 0
                        self.share_lock.acquire()
                        self.share_var.set(blog)
                        self.share_lock.release()

                        while True:
                            self.share_lock_4.acquire()
                            if "resultNumber" in self.share_var_4.keys() and "inputVideoPath" in self.share_var_4.keys() and "outputVideoPath" in self.share_var_4.keys() and self.share_var_4["resultNumber"] and self.share_var_4["inputVideoPath"] and self.share_var_4["outputVideoPath"]:
                                resultNumber = self.share_var_4["resultNumber"]
                                self.share_var_4["resultNumber"] = None
                                inputVideoPath = self.share_var_4["inputVideoPath"]
                                self.share_var_4["inputVideoPath"] = None
                                outputVideoPath = self.share_var_4["outputVideoPath"]
                                self.share_var_4["outputVideoPath"] = None
                            self.share_lock_4.release()
                            if resultNumber and inputVideoPath and outputVideoPath:
                                break
                        return '{"orderid": "' + orderid_str + '","code": "200","msg": "stop ok","data": {"resultNumber":"'+ resultNumber +'", "inputVideoPath":"'+ inputVideoPath +'", "outputVideoPath":"'+ outputVideoPath +'"}}'
                    elif order_str == 'getStatus':
                        gpuStatus = get_gpu_status()
                        gpuStatus = str(gpuStatus)
                        return '{"orderid": "' + orderid_str + '","code": "200","msg": "getStatus ok","data": {"modelStatusCode":"200", "gpuStatus": "'+ gpuStatus +'"}}'
                    else:
                        return 'no order'
                else:
                    return 'no order'
            else:
                return 'invalid order'
            # except TypeError:
            #     print('TypeError')
            #     return 'receive str TypeError'
            # except Exception:
            #     print('Exception')
            #     return 'receive str Exception'
        else:
            return 'json ValueError'



