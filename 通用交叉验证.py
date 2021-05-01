def tfidf(ngram,max_feature):
    return TfidfVectorizer(
        sublinear_tf = True,
        strip_accents = 'unicode',
        analyzer = 'word',
        token_pattern = r'\w{1,}',
        stop_words = 'english',
        ngram_range = (1,ngram),
        max_features = max_feature,
    )

# k 折交叉验证
def k_evaluate(clf,x_train,y_train,x_test,k=10,n_est=0):
    skf = StratifiedKFold(n_splits=k, random_state=7,shuffle=True) 
    test_preds = np.zeros((x_test.shape[0],1),int)
    for KF_index,(train_index,valid_index) in enumerate(skf.split(x_train, train_df['label'].values)):
        logging.info('第%d折交叉验证开始...'%(KF_index + 1))
        # 训练集划分
        x_train_,x_valid_ = x_train[train_index],x_train[valid_index]
        y_train_,y_valid_ = y_train[train_index],y_train[valid_index]
        # 开始训练...
        clf.fit(x_train_,y_train_)
        # 执行预测
        val_pred = clf.predict(x_valid_)
        logging.info('准确率为：%.7f'%f1_score(y_valid_,val_pred,average='macro'))
        test_preds = np.column_stack((test_preds,clf.predict(x_test)))
#         test_pred += clf.predict_proba(x_test)
        logging.info('保存模型est%d_KF_index%d'%(n_est,KF_index + 1))
        joblib.dump(clf,data_path + 'LGBM/model/est%d_KF_index%d'%(n_est,KF_index + 1),compress=3)
    return test_preds

def save_pred2file(saved_path,test_preds):
    preds = []
    for i,test_list in enumerate(test_preds):    
        #  取预测数最多的作为预测结果   
        preds.append(np.argmax(np.bincount(test_list)))
    preds = np.array(preds)
    submission = pd.DataFrame()
    submission['label'] = preds
    submission.to_csv(saved_path,index=False)
    print("保存完毕")   
