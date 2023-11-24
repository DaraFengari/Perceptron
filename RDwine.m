% Cargar los datos desde el archivo Wine.csv
data = csvread('Wine.csv');
X = data(:, 1:11)';  % Tomar las primeras 11 columnas como características
y = data(:, 12)';    % Tomar la última columna como etiquetas de clase

% Dividir los datos en conjuntos de entrenamiento y prueba
train_ratio = 0.8;
test_ratio = 0.2;
num_samples = size(X, 2);
num_train = round(train_ratio * num_samples);

X_train = X(:, 1:num_train);
y_train = y(1:num_train);
X_test = X(:, num_train+1:end);
y_test = y(num_train+1:end);

% Crear la red neuronal multicapa
hidden_layer_size = 10;
net = patternnet(hidden_layer_size);

% Configurar el conjunto de entrenamiento
net.divideFcn = 'divideind';  % Utilizar un conjunto de entrenamiento personalizado
net.divideParam.trainInd = 1:num_train;
net.divideParam.valInd = [];  % No utilizar conjunto de validación
net.divideParam.testInd = num_train+1:num_samples;

% Configurar hiperparámetros de entrenamiento
net.trainParam.epochs = 100;
net.trainParam.lr = 0.01;

% Entrenar la red neuronal
net = train(net, X_train, y_train);

% Realizar predicciones en el conjunto de prueba
y_pred = net(X_test);

% Convertir las salidas continuas a etiquetas binarias
y_pred_binary = round(y_pred);

% Calcular la precisión en el conjunto de prueba
accuracy = sum(y_pred_binary == y_test) / numel(y_test) * 100;
fprintf('Precisión en el conjunto de prueba: %.2f%%\n', accuracy);
