% Cargar los datos desde el archivo Wine.csv
data = csvread('Wine.csv');
X = data(:, 1:11)';
y = data(:, 12)';

% Inicializar variables para almacenar precisiones
accuracy_loo = zeros(size(y));
accuracy_lko = zeros(size(y));

% Leave-One-Out (LOO)
for i = 1:length(y)
    X_train_loo = X;
    y_train_loo = y;
    X_train_loo(:, i) = [];
    y_train_loo(i) = [];
    
    X_test_loo = X(:, i);
    y_test_loo = y(i);
    
    % Crear la red neuronal multicapa para LOO
    net_loo = patternnet(hidden_layer_size);
    net_loo.divideFcn = 'divideind';
    net_loo.divideParam.trainInd = 1:length(y_train_loo);
    net_loo.divideParam.valInd = [];
    net_loo.divideParam.testInd = length(y_train_loo) + 1;
    
    % Configurar hiperparámetros de entrenamiento
    net_loo.trainParam.epochs = 100;
    net_loo.trainParam.lr = 0.1;
    
    % Entrenar la red neuronal para LOO
    net_loo = train(net_loo, X_train_loo, y_train_loo);
    
    % Realizar predicciones en el conjunto de prueba para LOO
    y_pred_loo = net_loo(X_test_loo);
    
    % Convertir las salidas continuas a etiquetas binarias
    y_pred_binary_loo = round(y_pred_loo);
    
    % Calcular la precisión en el conjunto de prueba para LOO
    accuracy_loo(i) = y_pred_binary_loo == y_test_loo;
end

accuracy_loo_average = mean(accuracy_loo);
fprintf('Precisión usando Leave-One-Out (LOO): %.2f%%\n', accuracy_loo_average);

% Leave-K-Out (LKO) con k=5
k = 5;
num_samples = length(y);
indices_lko = zeros(num_samples, 1);

% Generar índices de particionamiento para LKO
for i = 1:k
    test_indices_lko = find(mod(1:num_samples, k) == (i - 1));
    indices_lko(test_indices_lko) = i;
end

for i = 1:k
    test_indices_lko = (indices_lko == i);
    train_indices_lko = ~test_indices_lko;

    X_train_lko = X(:, train_indices_lko);
    y_train_lko = y(train_indices_lko);
    
    X_test_lko = X(:, test_indices_lko);
    y_test_lko = y(test_indices_lko);
    
    % Crear la red neuronal multicapa para LKO
    net_lko = patternnet(hidden_layer_size);
    net_lko.divideFcn = 'divideind';
    net_lko.divideParam.trainInd = 1:length(y_train_lko);
    net_lko.divideParam.valInd = [];
    net_lko.divideParam.testInd = length(y_train_lko) + 1:length(y_train_lko) + length(y_test_lko);
    
    % Configurar hiperparámetros de entrenamiento
    net_lko.trainParam.epochs = 100;
    net_lko.trainParam.lr = 0.1;
    
    % Entrenar la red neuronal para LKO
    net_lko = train(net_lko, X_train_lko, y_train_lko);
    
    % Realizar predicciones en el conjunto de prueba para LKO
    y_pred_lko = net_lko(X_test_lko);
    
    % Convertir las salidas continuas a etiquetas binarias
    y_pred_binary_lko = round(y_pred_lko);
    
    % Calcular la precisión en el conjunto de prueba para LKO
    accuracy_lko(test_indices_lko) = y_pred_binary_lko == y_test_lko;
end

accuracy_lko_average = mean(accuracy_lko);
fprintf('Precisión usando Leave-K-Out (LKO) con k=%d: %.2f%%\n', k, accuracy_lko_average);
