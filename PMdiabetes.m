% Cargar los datos desde el archivo diabetes.csv
data = csvread('diabetes.csv');
X = data(:, 1:8)';
y = data(:, 9)';

% Crear la red neuronal multicapa
hidden_layer_size = 10;
net = patternnet(hidden_layer_size);

% Configurar hiperparámetros de entrenamiento
net.trainParam.epochs = 100;
net.trainParam.lr = 0.01;

% Inicializar variables para almacenar precisiones
accuracy_loo = zeros(size(y));
accuracy_lko = zeros(size(y));

% Leave-One-Out (LOO)
cv_loo = cvpartition(length(y), 'LeaveOut');

for i = 1:cv_loo.NumTestSets
    train_indices_loo = cv_loo.training(i);
    test_indices_loo = cv_loo.test(i);
    
    X_train_loo = X(:, train_indices_loo);
    y_train_loo = y(train_indices_loo);
    
    X_test_loo = X(:, test_indices_loo);
    y_test_loo = y(test_indices_loo);
    
    % Configurar el conjunto de entrenamiento para LOO
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = find(train_indices_loo);
    net.divideParam.valInd = [];
    net.divideParam.testInd = find(test_indices_loo);
    
    % Entrenar la red neuronal
    net = train(net, X_train_loo, y_train_loo);
    
    % Realizar predicciones en el conjunto de prueba
    y_pred_loo = net(X_test_loo);
    
    % Convertir las salidas continuas a etiquetas binarias
    y_pred_binary_loo = round(y_pred_loo);
    
    % Calcular la precisión en el conjunto de prueba
    accuracy_loo(test_indices_loo) = y_pred_binary_loo == y_test_loo;
end

accuracy_loo_percentage = mean(accuracy_loo) * 100;
fprintf('Precisión usando Leave-One-Out (LOO): %.2f%%\n', accuracy_loo_percentage);

% Leave-K-Out (LKO) con k=5
k = 5;
cv_lko = cvpartition(length(y), 'KFold', k);

for i = 1:cv_lko.NumTestSets
    train_indices_lko = cv_lko.training(i);
    test_indices_lko = cv_lko.test(i);
    
    X_train_lko = X(:, train_indices_lko);
    y_train_lko = y(train_indices_lko);
    
    X_test_lko = X(:, test_indices_lko);
    y_test_lko = y(test_indices_lko);
    
    % Configurar el conjunto de entrenamiento para LKO
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = find(train_indices_lko);
    net.divideParam.valInd = [];
    net.divideParam.testInd = find(test_indices_lko);
    
    % Entrenar la red neuronal
    net = train(net, X_train_lko, y_train_lko);
    
    % Realizar predicciones en el conjunto de prueba
    y_pred_lko = net(X_test_lko);
    
    % Convertir las salidas continuas a etiquetas binarias
    y_pred_binary_lko = round(y_pred_lko);
    
    % Calcular la precisión en el conjunto de prueba
    accuracy_lko(test_indices_lko) = y_pred_binary_lko == y_test_lko;
end

accuracy_lko_percentage = mean(accuracy_lko) * 100;
fprintf('Precisión usando Leave-K-Out (LKO) con k=%d: %.2f%%\n', k, accuracy_lko_percentage);
