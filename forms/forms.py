import quart.flask_patch

from flask_wtf import FlaskForm
from wtforms import TextAreaField, DecimalField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length


class InputForm(FlaskForm):
    """
    Fields for home page input.
    """
    model_type = SelectField('model_type', choices=[
        ('s', 'Sentence'), ('p', 'Paragraph')])
    input_text = TextAreaField(
        'input_text', [DataRequired(), Length(max=2000)])
    div_factor = DecimalField(
        'div_factor', [DataRequired()], places=1)
    submit = SubmitField("Go")
