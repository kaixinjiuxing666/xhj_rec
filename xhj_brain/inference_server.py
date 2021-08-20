#!/home/dell/anaconda3/envs/py38_imgsim/bin/python
import datetime
import json
import multiprocessing
import os
import time
from multiprocessing.managers import BaseManager
import psutil
import tornado.ioloop
import tornado.web
from map import mapping
from inference import run_inf

class compareHandler(tornado.web.RequestHandler):
    def post(self):
        start = datetime.datetime.now()  # 开始时间
        #print(self.request.body)
        try:
            url_json = json.loads(self.request.body,strict=False)
            # {
            #     "user_lst": [4100, 4300, 3, "恒小绿洲"]
            # }
            user_lst = url_json["user_lst"]

            response_dict = {}
            #print(user_lst)
            housr_value = run_inf(user_lst)

            response_dict["user  -----> house"] = housr_value

            #response_dict["error"] = housr_value
            self.finish(response_dict)
        except Exception as e:
            print(e)
            self.finish({"error": str(e)})

        end = datetime.datetime.now()  # 结束时间
        print("total_process_time:", (end - start).total_seconds())

def make_app():
    return tornado.web.Application([
        (r"/inference", compareHandler)
    ])


class SimpleClass(object):
    def __init__(self, port):
        self.port = port
        self.thread_tornado = None
        self.pid = -1

    def set(self):
        print("SimpleClass： set init Done")

    def run(self):
        return self.thread_tornado.start()

    def stop(self):
        return self.thread_tornado.stop()

    def set_pid(self, pid):
        self.pid = pid

    def get_pid(self):
        return self.pid


class Worker(multiprocessing.Process):
    def __init__(self, port, simple_class):
        multiprocessing.Process.__init__(self)

        self.log_writer = ""
        self.model_obj = ""
        self.port = port
        self.simple_class = simple_class

    def run(self):
        print("Worker： run() running")
        self.simple_class.set_pid(os.getpid())
        app = make_app()
        app.listen(self.port)
        tornado.ioloop.IOLoop.instance().start()

    def stop(self):
        self.terminate()


if __name__ == "__main__":

    port = 35899

    BaseManager.register('SimpleClass', SimpleClass)
    manager = BaseManager()
    manager.start()

    simple_class = manager.SimpleClass(port)

    tornado_server_process = None
    while True:

        if not psutil.pid_exists(simple_class.get_pid()):
            tornado_server_process = Worker(port, simple_class)
            tornado_server_process.start()
        else:
            pass
        time.sleep(5)  # 休眠5秒