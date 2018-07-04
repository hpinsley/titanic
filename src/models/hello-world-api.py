from flask import Flask, request
# If you have problems running in debug mode (errno = 2)
# see https://stackoverflow.com/questions/37650208/flask-cant-find-app-file
# python -m hello-world-api run

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def say_hello():
    data = request.get_json(force=True)
    name = data['name']
    return "hello {}".format(name)

if (__name__ == '__main__'):
    app.run(port=10001, debug=True)
