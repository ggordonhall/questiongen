from quart import Quart, render_template, request, redirect, url_for
from forms import InputForm

from decoding.pipeline import DecodingPipeline

"""
Route logic for the Quart webserver
"""


# INITIALISATION
# =====================================
application = Quart(__name__)
application.debug = True
application.secret_key = "My super secret key"


# ROUTES
# =====================================
@application.route('/', methods=['POST', 'GET'])
async def input():
    form = InputForm()
    if form.validate_on_submit():
        r_form = await request.form
        model_type = r_form['model_type']
        inp_text = r_form['input_text']
        div_factor = r_form['div_factor']
        return redirect(url_for('result', model_type=model_type, inp_text=inp_text, div_factor=float(div_factor)))
    return await render_template('homepage.html', form=form)


@application.route('/result')
async def result():
    model_type = request.args['model_type']
    inp_text = request.args['inp_text']
    div_factor = request.args['div_factor']
    pred = DecodingPipeline(model_type, inp_text, div_factor).pred
    if model_type == 'p':
        question, answer = pred
        pred = [question, '\n', 'Answer: ' + answer]
    else:
        pred = [pred]
    return await render_template('resultpage.html', inp_text=inp_text, question=pred)


if __name__ == '__main__':
    application.run()
