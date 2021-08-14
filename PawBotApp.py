from flask import Flask, render_template, jsonify, request
import PawBot

app = Flask(__name__)

app.config['SECRET_KEY'] = 'pawbotpetmedicalchatbot32715'

@app.route('/', methods=["GET"])
def pawbot_ui():
    return render_template('pawBotHTML.html', **locals())


@app.route('/dogselect', methods=["POST"])
def dog_selector():
    pet_type = 1
    PawBot.chose_dog()
    return jsonify({"type": pet_type})


@app.route('/catselect', methods=["POST"])
def cat_selector():
    pet_type = 2
    PawBot.chose_cat()
    return jsonify({"type": pet_type})


@app.route('/pawbot', methods=["POST"])
def chatbot_response():
    if request.method == 'POST':
        the_question = request.form['question']
        response = PawBot.pawbot_response(the_question)
    return jsonify({"answer": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)
