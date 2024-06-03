from preprocessing import preproc_train
from preprocessing import label_categorize


#preprocessing trainning data
X, y = preproc_train()
y_cat = label_categorize(y)
