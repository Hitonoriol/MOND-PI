import base64
from io import BytesIO
from flask import request


# Get a Float parameter with name `name` from the request or
# return the specified default value if it's absent
def num(name, default=0):
    val = request.args.get(name)
    return float(float(val) if val is not None else default)


# Get an Int parameter with name `name` from the request
def inum(name, default=0):
    return int(num(name, default))


# Get a string parameter with name `name` from the request
def arg(name):
    return request.args.get(name)


# Get a bool parameter with name `name` from the request
def bln(name, default=False):
    val = arg(name)
    return default if arg is None else arg == "true"


class Output:
    def __init__(self):
        self.contents = """
        <!DOCTYPE html><html>
        <head>
            <style>
                body {
                    font-size: 14px;
                    overflow-wrap: break-word;
                    word-wrap: break-word;
                }
                pre {
                    font-family: "Lucida Console", "Courier New", monospace;
                    display: inline;
                    width: 100%;
                    margin: 0;
                    white-space: pre-wrap;
                    white-space: -moz-pre-wrap;
                    white-space: -pre-wrap;
                    white-space: -o-pre-wrap;
                    word-wrap: break-word;
                }
                img {
                    max-width: 100%;
                }
            </style>
        </head
        <body>
        """

    def out(self, text="", end="<br>"):
        if not type(text) is str:
            text = str(text)

        text = text.replace('\n', '<br>')
        self.contents += f"<pre>{text}{end}</pre>"
        return self

    def plot(self, fig, end="<br>"):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        self.contents += f"<img src='data:image/png;base64,{data}'/>{end}"
        return self

    def get(self):
        return f"{self.contents}</body></html>"
