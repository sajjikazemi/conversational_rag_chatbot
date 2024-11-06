import modal
from hello import app, hello

with app.run(show_progress=False):
    #reply = hello.local()
    reply=hello.remote()

print(reply)