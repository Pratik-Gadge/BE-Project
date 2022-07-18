import os

from flask import Flask,flash,redirect, request, render_template, abort, url_for
app = Flask(__name__)

import PyPDF2
from PyPDF2.pdf import *  # to import function used in origimal `extractText`
from styleclass.classify import classify
from styleclass.train import get_default_model


app.secret_key="your secret key"

UPLOAD_FOLDER = "C:/Users/Sachin/PycharmProjects/Flask-app/static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

fileloc=''

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello():
    return render_template("index.html")



@app.route("/" , methods = ['GET','POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        fileloc =  app.config['UPLOAD_FOLDER']+f.filename

        def myExtractText(self, distance=None):
            # original code from `page.extractText()`
            # https://github.com/mstamy2/PyPDF2/blob/d7b8d3e0f471530267827511cdffaa2ab48bc1ad/PyPDF2/pdf.py#L2645

            text = u_("")

            content = self["/Contents"].getObject()

            if not isinstance(content, ContentStream):
                content = ContentStream(content, self.pdf)

            prev_x = 0
            prev_y = 0

            for operands, operator in content.operations:
                # used only for test to see values in variables
                # print('>>>', operator, operands)

                if operator == b_("Tj"):
                    _text = operands[0]
                    if isinstance(_text, TextStringObject):
                        text += _text
                elif operator == b_("T*"):
                    text += "\n"
                elif operator == b_("'"):
                    text += "\n"
                    _text = operands[0]
                    if isinstance(_text, TextStringObject):
                        text += operands[0]
                elif operator == b_('"'):
                    _text = operands[2]
                    if isinstance(_text, TextStringObject):
                        text += "\n"
                        text += _text
                elif operator == b_("TJ"):
                    for i in operands[0]:
                        if isinstance(i, TextStringObject):
                            text += i
                    text += "\n"

                if operator == b_("Tm"):

                    if distance is True:
                        text += '\n'

                    elif isinstance(distance, int):
                        x = operands[-2]
                        y = operands[-1]

                        diff_x = prev_x - x
                        diff_y = prev_y - y

                        # print('>>>', diff_x, diff_y - y)
                        # text += f'| {diff_x}, {diff_y - y} |'

                        if diff_y > distance or diff_y < 0:  # (bigger margin) or (move to top in next column)
                            text += '\n'
                            # text += '\n' # to add empty line between elements

                        prev_x = x
                        prev_y = y

            return text

        # --- main ---

        pdfFileObj = open( fileloc , 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        text = ''

        for page in pdfReader.pages:
            # text += page.extractText()  # original function
            # text += myExtractText(page)        # modified function (works like original version)
            # text += myExtractText(page, True)  # modified function (add `\n` after every `Tm`)
            text += myExtractText(page, 17)  # modified function (add `\n` only if distance is bigger then `17`)

        # get only text after word `References`
        pos = text.lower().find('references')

        # only referencers as text
        references = text[pos + len('references '):]

        # doc without references
        doc = text[:pos]

        # referencers as list
        references = references.split('\n')

        # remove empty lines and lines which have 2 chars (ie. page number)
        references = [item.strip() for item in references if len(item.strip()) > 2]

        print('\n--- names ---\n')

        data = []

        model = get_default_model()

        #dict = {}

        for nubmer, line in enumerate(references, 1):  # skip last element with page number
            line = line.strip()
            # number = 0
            if line:  # skip empty line
                for char in line:
                    if char in "~?!/^":
                        line.replace(char, '')
                authors_and_year = re.match('((.*)\. (\d{4})\.)', line)
                text, authors, year = authors_and_year.groups()
                # print(text, '|', authors, '|', year)
                names = re.split(',[ ]*and |,[ ]*| and ', authors)
                # print(names)
                #dict[number] = {}
                #dict[number]['line'] = line
                # [(name, last_name), ...]
                names = [(name, name.split(' ')[-1]) for name in names]
                # print(names)

                # print(' line:', line)
                # print('   text:', text)
                #dict[number]['text'] = text
                # print('authors:', authors)
                #dict[number]['authors'] = authors
                # print('   year:', year)
                #dict[number][year] = year
                # print('  names:', names)
                for char in line:
                    if char in "~?!/-":
                        line.replace(char, '')
                prediction = classify(line, *model)
                cite = prediction[0].upper()
                # print("Citation Style :", cite.upper())
                # print('---')
                #dict[number]['citation'] = cite.upper()
                data.append([line, authors, year, cite])
                #number = number + 1

        #for i in data:
        #    for j in i:
        #        print("\n", j)

        return render_template("index.html", data=data)


@app.route("/citation_analysis")
def output():
    return render_template("citation_analysis.html")

@app.route("/citation_analysis", methods = ['GET','POST'])
def citation():
    if request.method == 'POST':
        cite = request.form.get('Citation')
        model = get_default_model()
        prediction = classify(cite , *model)
        citation = prediction[0].upper()
        return render_template("citation_analysis.html" , content = citation , cite = cite)

if __name__ == '__main__':
    app.run(debug=True)