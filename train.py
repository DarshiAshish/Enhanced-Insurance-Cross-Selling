import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_data["Gender"])
train_data["Gender"] = le.transform(train_data["Gender"])
test_data["Gender"] = le.transform(test_data["Gender"])


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_data["Vehicle_Age"])
train_data["Vehicle_Age"] = le.transform(train_data["Vehicle_Age"])
test_data["Vehicle_Age"] = le.transform(test_data["Vehicle_Age"])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_data["Vehicle_Damage"])
train_data["Vehicle_Damage"] = le.transform(train_data["Vehicle_Damage"])
test_data["Vehicle_Damage"] = le.transform(test_data["Vehicle_Damage"])

test_data.drop(["id"],axis=1,inplace=True)
train_data.drop(["id"],axis=1,inplace=True)

xtrain = train_data.drop(["Response"],axis=1)
ytrain = train_data[["Response"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.3, stratify=ytrain, random_state=42)


def model_metrics(model, test_features, test_labels):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score,f1_score,recall_score
    out=model.predict(test_features)
    acc = accuracy_score(out,test_labels)*100
    pre = precision_score(out,test_labels,average = 'weighted')*100
    f1 = f1_score(out,test_labels,average = "weighted")*100
    re = recall_score(out,test_labels,average="weighted")*100
    print("Accuracy : ",acc, flush=True)
    print("Precision : ",pre, flush=True)
    print("f1_score : ",f1, flush=True)
    print("recall score : ",re, flush=True)
    print("---------------------------------------------------------------------------", flush=True)
    return acc, pre, f1, re


def random_forest_model(train_features, train_labels, test_features, test_labels, input_params={}, cv=5):
    from sklearn.ensemble import RandomForestClassifier
    rc = RandomForestClassifier()
    rc.fit(train_features,train_labels)
    print("-----------------------------------------------------------------------------", flush=True)
    print("Random Forest model", flush=True)
    acc, pre, f1, re = model_metrics(rc, test_features, test_labels)
    return rc, acc, pre, f1, re


regr, acc, pre, f1, re = random_forest_model(x_train, y_train, x_test, y_test)

with open("metrics.txt", 'w') as outfile:
    outfile.write("Accuracy: %2.1f%%\n" % acc)
    outfile.write("Precision: %2.1f%%\n" % pre)
    outfile.write("F1 Score: %2.1f%%\n" % f1)
    outfile.write("Recall Score: %2.1f%%\n" % re)

importances = regr.feature_importances_
labels = train_data.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)
    
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()



