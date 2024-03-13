import numpy
import numpy as np
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import copy


from keras import backend as K
from keras.layers import Layer
from keras.initializers import RandomUniform, Initializer, Constant
from keras.layers import Activation, Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Kriging:

    def __init__(self, cluster = None):
            self.create_model_from_cluster(cluster)
    def create_model_from_cluster(self,cluster):
        self.scaler = preprocessing.StandardScaler()
        cluster = np.array(cluster)
        X = cluster[:, 0:16] # features from 0 to 15th index
        y = cluster[:, 16:22] # value at 3th index

        X = self.scaler.fit_transform(X)
        kernel = 1.0 * RBF(0.5)  # squared-exponential kernel
        self.model = GaussianProcessRegressor(kernel=kernel, random_state=0, copy_X_train=False).fit(X, y)


    def test(self, cluster):
        mae = 0
        for i in range(len(cluster)):
            y_act = cluster[i][16]
            Y_pred = self.predict(cluster[i][:16])

            mae = mae + abs(y_act - Y_pred)
        self.mae = mae/len(cluster)
    
    def predict(self, value):
        value = numpy.array([value])
        B = np.reshape(value, (1, 16))
        B= (self.scaler.transform(B))
        y_pred = self.model.predict(B)

        return  y_pred[0]


class Polynomial_Regression:

    def __init__(self, degree=-1, index =-1,filename='',cluster = None):
        # if cluster == None:
        #     self.train(degree,index,filename)
        # else:
        #     self.create_model_from_cluster(cluster,degree)
        self.create_model_from_cluster(cluster,degree)


    def create_model_from_cluster(self,cluster,deg):
        self.scaler = preprocessing.StandardScaler()
        cluster = np.array(cluster)

        X = cluster[:, 0:16] # features from 0 to 15th index
        y = cluster[:, 16:22] # value at 16th index
        y[y < 0] = 0
        y[y > 1] = 1

        X = self.scaler.fit_transform(X)
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        self.poly_reg = PolynomialFeatures(degree=deg)
        X_poly = self.poly_reg.fit_transform(X)
        self.pol_reg = LinearRegression(copy_X=False)
        self.pol_reg.fit(X_poly, y)

    def test(self, cluster):
        mae = 0
        for i in range(len(cluster)):
            y_act = cluster[i][16]
            Y_pred = self.predict(cluster[i][:16])

            mae = mae + abs(y_act - Y_pred)
        self.mae = mae/len(cluster)
        
    def predict(self, value):
        value = numpy.array([value])
        B = np.reshape(value, (1, 16))
        B= self.scaler.transform(B)

        y_pred = self.pol_reg.predict(self.poly_reg.fit_transform(B))
        return y_pred[0]


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)



    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

    def gaussian(self, x, mu, sigma):
        return exp(- metrics(mu, x) ** 2 / (2 * sigma ** 2))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Model:
    ss = StandardScaler()

    def __init__(self,no_of_neurons,cluster):
        
        self.train(no_of_neurons,np.array(cluster))
    def train(self,no_of_neurons,cluster):
        # dataset = pd.read_csv(clean_file_name,header=None)
        # for x in cluster:
        #     print(x)
        X = cluster[:, 0:16]
        y = cluster[:, 16:17]
        y[y < 0] = 0
        y[y > 1] = 1

        X = self.ss.fit_transform(X)
        self.model = Sequential()
        rbflayer = RBFLayer(no_of_neurons,
                                initializer=InitCentersRandom(X),
                                betas=3.0,
                                input_shape=(16,))
        self.model.add(rbflayer)
        self.model.add(Dense(1))
        self.model.add(Activation('linear'))
        self.model.compile(loss='mean_absolute_error',
                          optimizer = 'adam')
        history = self.model.fit(X, y, epochs=1000, batch_size=8,verbose=0)

    def test(self, cluster):
        mae = 0
        for i in range(len(cluster)):
            y_act = cluster[i][16]
            Y_pred = self.predict(cluster[i][:16])
            if y_act > 1:
                y_act =1
            if y_act < 0:
                y_act =0
            mae = mae + abs(y_act - Y_pred)
        self.mae = mae / len(cluster)

    def predict(self,val):
        value = np.array([val])
        B = self.ss.transform(value)
        y_pred = self.model.predict([B], verbose=0)

        if y_pred[0][0] > 1:
            return 1
        if y_pred[0][0] < 0:
            return 0
        return y_pred[0][0]
    

class Ensemble:
    def __init__(self,database,deg=2):

        database = database[["road_type", "road_id", "scenario_length", "vehicle_front", "vehicle_adjacent", 
                             "vehicle_opposite", "vehicle_front_two_wheeled", "vehicle_adjacent_two_wheeled", 
                             "vehicle_opposite_two_wheeled", "time", "weather", "pedestrian_density", 
                             "target_speed", "trees", "buildings", "task", "follow_center",
                             "avoid_vehicles", "avoid_pedestrians", "avoid_static", "abide_rules", 
                             "reach_destination"]].values.tolist()

        self.vars = ["road_type", "road_id", "scenario_length", "vehicle_front", "vehicle_adjacent", 
                             "vehicle_opposite", "vehicle_front_two_wheeled", "vehicle_adjacent_two_wheeled", 
                             "vehicle_opposite_two_wheeled", "time", "weather", "pedestrian_density", 
                             "target_speed", "trees", "buildings", "task"]
        # database.columns = range(4)

        train, test = train_test_split(database, test_size=0.2)
        self.rbf = Model(10, train)
        self.PR = Polynomial_Regression(degree=deg, cluster = train)
        self.KR = Kriging(train)

        self.rbf.test(test)
        self.PR.test(test)
        self.KR.test(test)



        total_mae = self.rbf.mae + self.PR.mae + self.KR.mae
        # total_mae = self.PR.mae + self.KR.mae
        self.w_rbf = 0.5 * ((total_mae - self.rbf.mae)/total_mae)
        self.w_PR = 0.5 * ((total_mae - self.PR.mae) / total_mae)
        self.w_KR = 0.5 * ((total_mae - self.KR.mae) / total_mae)




    def predict (self,fv):
        # fv = fv[:16]
        # fv = [0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 3, 1, 4, 1, 1, 1]
        # print(fv)
        y_rbf = self.rbf.predict(copy.deepcopy(fv))
        y_pr = self.PR.predict(copy.deepcopy(fv))
        y_kr = self.KR.predict(copy.deepcopy(fv))

        # print(y_pr)
        # print((y_rbf*self.w_rbf) + (y_pr*self.w_PR) + (y_kr*self.w_KR))

        
        # print(diff_pr_kr,diff_rbf_kr,diff_rbf_pr)

        # pred_high = (y_rbf[0]*self.w_rbf[0]) + (y_pr[0]*self.w_PR[0]) + (y_kr[0]*self.w_KR[0])
        # pred_low = (y_rbf[1]*self.w_rbf[1]) + (y_pr[1]*self.w_PR[1]) + (y_kr[1]*self.w_KR[1])

        # pred_high = (y_pr[0]*self.w_PR[0]) + (y_kr[0]*self.w_KR[0])
        # pred_low = (y_pr[1]*self.w_PR[1]) + (y_kr[1]*self.w_KR[1])
        # return max(pred_high - 200, 50 - pred_low)
        
        

        return min((y_rbf*self.w_rbf) + (y_pr*self.w_PR) + (y_kr*self.w_KR))

