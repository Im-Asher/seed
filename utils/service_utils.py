
from utils.constant_enum import ResponseCode


class Response:
    def __init__(self, 
                 status=ResponseCode.Fail.value, 
                 results=None, 
                 msg=None, 
                 original=None):
        self.status = status
        self.results = results
        self.msg = msg
        self.original = original
