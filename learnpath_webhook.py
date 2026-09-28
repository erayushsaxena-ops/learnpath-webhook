import base64
from IPython.display import HTML, display

def make_link(filepath, filename):
    with open(filepath, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{data}" download="{filename}">{filename}</a>'

display(HTML(make_link("learnpath_webhook.py", "learnpath_webhook.py")))
