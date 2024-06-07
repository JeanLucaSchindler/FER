from logic_ml.preprocessing import preproc_train, get_image, preproc
from logic_ml.preprocessing import label_categorize
from logic_ml.ResNet50V2 import load_model, add_last_layers, finetuning, callbacks, fit_model, eval_model, predict_model
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('/Users/marinelegall/code/marinoulg/FER_Project/FER/raw_data/labels.csv')

    path = 'raw_data/'+data['pth']

    get_image(path)


    #preprocessing trainning data
    X_train, y_train = preproc_train()
    y_cat = label_categorize(y_train)

    preproc(path)

    # Loading model
    model = load_model(X_train)

    # Adding last layers
    model = add_last_layers(model)

    model.summary()

    # compile and finetuning model
    model = finetuning(model, LR=0.001)

    #callbacks = EarlyStopping and LReduceOnPlateau
    my_callbacks = callbacks()

    fit_model(model, my_callbacks)

    eval_model(model)

    predict_model(model)
