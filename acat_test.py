import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
#class cliassify():
ds = pd.read_csv('completeData.csv')
headers = ['business_classification', 'business_impact', 'business_relevance',
       'category_id', 'data_confidentiality', 'eoldriver', 'extensibility',
       'geographical_scope', 'has_dependencies', 'id', 'io_intensity',
       'latency_sensitivity', 'no_of_users', 'scalability', 'service_level',
       'source_code_available', 'stage', 'type', 'user_facing',
       'workload_variation', 'dependencies.Hardware.Dependent',
       'dependencies.Operating.Environment.Dependent',
       'dependencies.Operating.System.Dependent', 'IsPassPlatAvail',
       'IsHarwareSupported', 'IsOSSupported', 'IsPlatformSupported',
       'IsDatabaseSupported']

categories={}
for f in headers:
    ds[f] = ds[f].astype('category')
    categories[f] = ds[f].cat.categories
#df_ = df[headers]
labels = pd.read_csv('completeData.csv', usecols = ['pivot.disposition_1'])
#print(labels)
#covert strings into numericals
df_ohe = pd.get_dummies(ds, columns = headers,dummy_na=True)
#print(df)
df_num, df_labels = pd.factorize(labels['pivot.disposition_1'])
print(df_num)
df_train, df_test = train_test_split(stack, test_size=0.25)

num_train_entries = df_train.shape[0]
num_train_features = df_train.shape[1]-1

num_test_entries = df_test.shape[0]
num_test_features = df_test.shape[1]-1

#create temp csv files
df_train.to_csv('train_temp.csv', index=False)
df_test.to_csv('test_temp.csv', index=False)
train_x = train_temp.csv(training[:,0])
train_y = test_temp.csv(training[:,1])
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(train_x, train_y, n_epoch=self.epoch, batch_size=8, show_metric=True)
model.save('/home/ubuntu/AIAP_experiments/ABCNN_v1/tp_model/model_Topic.tflearn')
 pickle.dump( {  'train_x':train_x, 'train_y':train_y}, open( "/home/Desktop/acat_code_run/acat_code_v3/Path_training_data", "wb" ) )
#feature_columns = [tf.contrib.layers.real_valued_column("", dimension=28)]
#model_directory = 'C:\Users\A664120\Desktop\acat_code_run\acat_code_v3\model_dir'
#model_file_name = '%s/model.pkl' % model_directory
#model_columns_file_name = '%s/model_columns.pkl' % model_directory
#print("model directory = %s" % model_dir)
#classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], \
 #                                           n_classes=5, model_dir=model_dir)


#from sklearn.externals import joblib
#joblib.dump(classifier, 'model.pkl')
#classifier = joblib.load('model.pkl')

