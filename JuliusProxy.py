import socket
import re

class JuliusProxy:
    """ JuliusProxy. Connects to julius and parses the recognition result."""
    def __init__(self, host="localhost", port=10500):
        """ Initialize the proxy. connect to julius -module."""
        self.sock = socket.socket(socket.AF_INET,
                                  socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.pattern = re.compile('([A-Z]+=".*?")')

    def getResult(self):
        """ Receive result as XML format."""
        msg = []
        # receive all messages
        while True:
            msg.append(str(self.sock.recv(1024)))
            if "</RECOGOUT>" in msg[-1]:
                break
        # connect them all, and split with \n
        self.msg = "".join([m.replace(".\n","") for m in msg])
        self.msg = self.msg.split("\n")
        return self.msg

    def parseResult(self):
        """ run after getResult. it parses the reseult
        and returns a dictionary having results"""

        # parse all WHYPO tags
        result = []
        for msg in [m for m in self.msg if "WHYPO" in m]:
            result.append({})

            for prop in self.pattern.findall(msg):
                key = prop.split("=")[0]
                value = prop.split('"')[1]

                if key == "CM":
                    value = float(value)
                if key == "CLASSID":
                    value = int(value)
                result[-1][key] = value

        return result

if __name__ == "__main__":
    proxy = JuliusProxy()
    while True:
        list = "\n".join(proxy.getResult())
        list2 = []
        print( "[")
        for result in proxy.parseResult():
            word = result['WORD']

            res = word[:-1]
            list2.append(res.strip())
        print(''.join(list2).strip('&lt;s&gt;'))
        print("]")

