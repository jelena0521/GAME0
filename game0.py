import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import catboost as cb
#path='/cos_person/uci/'
path=''
train_data=pd.read_csv(path+'input/train_set.csv')
test_data=pd.read_csv(path+'input/test_set.csv')

lbl = LabelEncoder()
features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan','contact','day','month','duration','campaign','pdays','previous','poutcome']

for i in features:
    for j in features:
        if i != j :
            train_data[i + j]=train_data[i].astype(str)+train_data[j].astype(str)
            test_data[i + j] = test_data[i].astype(str)+test_data[j].astype(str)
            # for k in features:
            #     if i !=k and j !=k:
            #         train_data[i+j+k] = train_data[i].astype(str) + train_data[j].astype(str)+train_data[k].astype(str)
            #         test_data[i+j+k] = test_data[i].astype(str) + test_data[j].astype(str) +test_data[k].astype(str)



cat_col1 = [i for i in train_data.select_dtypes(object).columns if i not in ['ID','y']]
cat_col2 = [i for i in test_data.select_dtypes(object).columns if i not in ['ID']]
# cat_col3=[i for i in train_data.columns if i not in ['ID','y','age', 'job', 'marital', 'education', 'default', 'housing', 'loan','contact','day','month','duration','campaign','pdays','previous','poutcome']]
# cat_col4=[i for i in train_data.columns if i not in ['id','age', 'job', 'marital', 'education', 'default', 'housing', 'loan','contact','day','month','duration','campaign','pdays','previous','poutcome']]
cat_col3= [i for i in train_data.columns if i not in ['ID','y']]
cat_col4 = [i for i in test_data.columns if i not in ['ID']]
for i in cat_col1:
    train_data[i] = lbl.fit_transform(train_data[i].astype(str))
for i in cat_col3:
    train_data['count_' + i] = train_data.groupby([i])[i].transform('count')
for i in cat_col2:
    test_data[i]=lbl.fit_transform(test_data[i].astype(str))
for i in cat_col4:
    test_data['count_' + i] = test_data.groupby([i])[i].transform('count')



#     train_data['count_1' + i] = train_data.groupby([i])[i].transform('count')
#     test_data['count_1' + i] = test_data.groupby([i])[i].transform('count')
print(train_data.head())
# print(train_data.corr())

model = lgb.LGBMClassifier(
        boosting_type="gbdt", num_leaves=30, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=10000, objective='binary',metric= 'auc',
    subsample=0.95, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.001, random_state=2017)
#model=cb.CatBoostClassifier()


from sklearn.model_selection import KFold
n_splits=10
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
feats1= [i for i in train_data.columns if i not in ['ID','y']]
feats2= [i for i in test_data.columns if i not in ['ID']]
train_x = train_data[feats1]
train_y = train_data['y']
test_x=test_data[feats2]

test_data['pred'] = 0
for train_idx, val_idx in kfold.split(train_x):
    train_x1 = train_x.loc[train_idx]
    train_y1 = train_y.loc[train_idx]
    test_x1 = train_x.loc[val_idx]
    test_y1 = train_y.loc[val_idx]
    #,(vali_x,vali_y)
    model.fit(train_x1, train_y1,eval_set=[(train_x1, train_y1),(test_x1, test_y1)],eval_metric='auc')
# print(model.best_score_)
    test_data['pred'] += model.predict_proba(test_x)[:,1]
test_data['pred'] = test_data['pred']/10
# train_data.rename(columns={'y':'pred'}, inplace = True)
# data_all=train_data.append(test_data,ignore_index=True)
test_data.to_csv('game0.csv')
#print(data_all.head())
