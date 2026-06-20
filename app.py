from flask import Flask, request, render_template
import pickle


# creating flask app
app = Flask(__name__)


# python main
if __name__ == '__main__':
    app.run(debug = True)