#!flask/bin/python
from app import app
from app.models import *
app.run(debug=True,host='0.0.0.0', port=8082)
