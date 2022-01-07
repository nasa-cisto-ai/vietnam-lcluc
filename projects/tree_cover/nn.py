
def nn():
    model = Sequential(name='Pavia_University')

    model.add(Input(shape = X_train[0].shape, name = 'Input_Layer'))

    model.add(BatchNormalization(name = 'BatchNormalization'))

    model.add(Dense(units = 128, activation= 'relu', name = 'Layer1'))
    model.add(Dense(units = 128, activation= 'relu', name = 'Layer2'))
    model.add(Dense(units = 128, activation= 'relu', name = 'Layer3'))
    model.add(Dense(units = 128, activation= 'relu', name = 'Layer4'))

    model.add(Dropout(rate = 0.2, name = 'Dropout1',))

    model.add(Dense(units = 64, activation= 'relu', name = 'Layer5'))
    model.add(Dense(units = 64, activation= 'relu', name = 'Layer6'))
    model.add(Dense(units = 64, activation= 'relu', name = 'Layer7'))
    model.add(Dense(units = 64, activation= 'relu', name = 'Layer8'))

    model.add(Dropout(rate = 0.2, name = 'Dropout2'))

    model.add(Dense(units = 32, activation= 'relu', name = 'Layer9'))
    model.add(Dense(units = 32, activation= 'relu', name = 'Layer10'))
    model.add(Dense(units = 32, activation= 'relu', name = 'Layer11'))
    model.add(Dense(units = 32, activation= 'relu', name = 'Layer12'))

    model.add(Dense(units = y_train.shape[1], activation= 'softmax', name = 'Output_Layer'))

    return model

if __name__ == "__main__":

    model = nn()
    model.summary()