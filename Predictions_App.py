from flask import Flask
from flask import url_for
from flask import redirect
from flask import render_template
from flask import request

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import shap
import xgboost as xgb
import io
import base64

app = Flask(__name__)

#load the model
model = xgb.Booster(model_file='xgb.model')


@app.route('/')
@app.route('/home')
def home():
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        list_of_inputs = [i for i in request.form.values()]
        
        #create a list of column names
        column_names = ['Temperature',
                        'Fuel_Price',
                        'MarkDown1',
                        'MarkDown2',
                        'MarkDown3',
                        'MarkDown4',
                        'MarkDown5',
                        'CPI',
                        'Unemployment',
                        'IsHoliday',
                        'Size',
                        'Store_1',
                        'Store_2',
                        'Store_3',
                        'Store_4',
                        'Store_5',
                        'Store_6',
                        'Store_7',
                        'Store_8',
                        'Store_9',
                        'Store_10',
                        'Store_11',
                        'Store_12',
                        'Store_13',
                        'Store_14',
                        'Store_15',
                        'Store_16',
                        'Store_17',
                        'Store_18',
                        'Store_19',
                        'Store_20',
                        'Store_21',
                        'Store_22',
                        'Store_23',
                        'Store_24',
                        'Store_25',
                        'Store_26',
                        'Store_27',
                        'Store_28',
                        'Store_29',
                        'Store_30',
                        'Store_31',
                        'Store_32',
                        'Store_33',
                        'Store_34',
                        'Store_35',
                        'Store_36',
                        'Store_37',
                        'Store_38',
                        'Store_39',
                        'Store_40',
                        'Store_41',
                        'Store_42',
                        'Store_43',
                        'Store_44',
                        'Store_45',
                        'Dept_1',
                        'Dept_2',
                        'Dept_3',
                        'Dept_4',
                        'Dept_5',
                        'Dept_6',
                        'Dept_7',
                        'Dept_8',
                        'Dept_9',
                        'Dept_10',
                        'Dept_11',
                        'Dept_12',
                        'Dept_13',
                        'Dept_14',
                        'Dept_16',
                        'Dept_17',
                        'Dept_18',
                        'Dept_19',
                        'Dept_20',
                        'Dept_21',
                        'Dept_22',
                        'Dept_23',
                        'Dept_24',
                        'Dept_25',
                        'Dept_26',
                        'Dept_27',
                        'Dept_28',
                        'Dept_29',
                        'Dept_30',
                        'Dept_31',
                        'Dept_32',
                        'Dept_33',
                        'Dept_34',
                        'Dept_35',
                        'Dept_36',
                        'Dept_37',
                        'Dept_38',
                        'Dept_39',
                        'Dept_40',
                        'Dept_41',
                        'Dept_42',
                        'Dept_43',
                        'Dept_44',
                        'Dept_45',
                        'Dept_46',
                        'Dept_47',
                        'Dept_48',
                        'Dept_49',
                        'Dept_50',
                        'Dept_51',
                        'Dept_52',
                        'Dept_54',
                        'Dept_55',
                        'Dept_56',
                        'Dept_58',
                        'Dept_59',
                        'Dept_60',
                        'Dept_65',
                        'Dept_67',
                        'Dept_71',
                        'Dept_72',
                        'Dept_74',
                        'Dept_77',
                        'Dept_78',
                        'Dept_79',
                        'Dept_80',
                        'Dept_81',
                        'Dept_82',
                        'Dept_83',
                        'Dept_85',
                        'Dept_87',
                        'Dept_90',
                        'Dept_91',
                        'Dept_92',
                        'Dept_93',
                        'Dept_94',
                        'Dept_95',
                        'Dept_96',
                        'Dept_97',
                        'Dept_98',
                        'Dept_99',
                        'Type_A',
                        'Type_B',
                        'Type_C',
                        'Month_1',
                        'Month_2',
                        'Month_3',
                        'Month_4',
                        'Month_5',
                        'Month_6',
                        'Month_7',
                        'Month_8',
                        'Month_9',
                        'Month_10',
                        'Month_11',
                        'Month_12',
                        'Week_1',
                        'Week_2',
                        'Week_3',
                        'Week_4',
                        'Week_5',
                        'Week_6',
                        'Week_7',
                        'Week_8',
                        'Week_9',
                        'Week_10',
                        'Week_11',
                        'Week_12',
                        'Week_13',
                        'Week_14',
                        'Week_15',
                        'Week_16',
                        'Week_17',
                        'Week_18',
                        'Week_19',
                        'Week_20',
                        'Week_21',
                        'Week_22',
                        'Week_23',
                        'Week_24',
                        'Week_25',
                        'Week_26',
                        'Week_27',
                        'Week_28',
                        'Week_29',
                        'Week_30',
                        'Week_31',
                        'Week_32',
                        'Week_33',
                        'Week_34',
                        'Week_35',
                        'Week_36',
                        'Week_37',
                        'Week_38',
                        'Week_39',
                        'Week_40',
                        'Week_41',
                        'Week_42',
                        'Week_43',
                        'Week_44',
                        'Week_45',
                        'Week_46',
                        'Week_47',
                        'Week_48',
                        'Week_49',
                        'Week_50',
                        'Week_51',
                        'Week_52',
                        'Year_2010',
                        'Year_2011',
                        'Year_2012']
        
        #create an array of zeros of shape (1, 207)
        base_array = np.zeros((1, 207))
        
        #fill in default values for temp, fuel_price, CPI, Unemployment, IsHoliday, Year_2012
        base_array[0][0] = 60
        base_array[0][1] = 3
        base_array[0][7] = 212
        base_array[0][8] = 7
        base_array[0][9] = False
        base_array[0][206] = 1
        
        #fill in size (index 10)
        base_array[0][10] = float(list_of_inputs[3])
        
        #get column index of each inputted value, and change that index position of base_array to 1
        for i in [0, 1, 2, 4]:
            var = list_of_inputs[i]
            index = column_names.index(var)
            base_array[0][index] = 1
        
        
        #convert array to a DMatrix
        matrix= xgb.DMatrix(base_array)
        
        #predict
        y_pred = float(model.predict(matrix))
        
        #create force plot
        explainer = shap.TreeExplainer(model)
        shap_value = explainer.shap_values(base_array)
        force_plot = shap.force_plot(explainer.expected_value, shap_value, base_array, feature_names=column_names, matplotlib=True, show=False)
        
        #convert plot to PNG
        pngImage = io.BytesIO()
        FigureCanvas(force_plot).print_png(pngImage)
        
        #encode PNG to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')


        return render_template('prediction.html', prediction='Estimated Weekly Sales: ${:.2f}'.format(y_pred), force_plot=pngImageB64String)

#run the app
if __name__ == '__main__':
    app.run(debug=True)