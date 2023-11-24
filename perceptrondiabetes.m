% Carga los datos desde el archivo diabetes.csv
data = csvread('diabetes.csv');

% Divide los datos en características (X) y etiquetas (y)
X = data(:, 1:8);
y = data(:, 9);

% Define la proporción de entrenamiento (80%) y prueba (20%)
train_ratio = 0.8;
test_ratio = 0.2;

% Divide los datos en conjuntos de entrenamiento y prueba
num_samples = size(X, 1);
num_train = round(train_ratio * num_samples);
num_test = num_samples - num_train;

X_train = X(1:num_train, :);
y_train = y(1:num_train);
X_test = X(num_train+1:end, :);
y_test = y(num_train+1:end);

% Inicializa los pesos y el sesgo del perceptrón
num_features = size(X, 2);
weights = rand(1, num_features);
bias = 3;

% Hiperparámetros
learning_rate = 0.01;
epochs = 10000;

% Entrenamiento del perceptrón
for epoch = 1:epochs
    for i = 1:num_train
        input = X_train(i, :);
        target = y_train(i);
        
        % Calcula la salida del perceptrón
        output = sum(input .* weights) + bias;
        
        % Aplica la función de activación (en este caso, una función escalón)
        if output >= 0
            prediction = 1;
        else
            prediction = 0;
        end
        
        % Calcula el error
        error = target - prediction;
        
        % Actualiza los pesos y el sesgo
        weights = weights + learning_rate * error * input;
        bias = bias + learning_rate * error;
    end
end

% Prueba del perceptrón
correct_predictions = 0;
for i = 1:num_test
    input = X_test(i, :);
    target = y_test(i);
    
    output = sum(input .* weights) + bias;
    
    if output >= 0
        prediction = 1;
    else
        prediction = 0;
    end
    
    if prediction == target
        correct_predictions = correct_predictions + 1;
    end
end

% Calcula la precisión del perceptrón en el conjunto de prueba
accuracy = correct_predictions / num_test * 100;

fprintf('Precisión en el conjunto de prueba: %.2f%%\n', accuracy);

% Gráfico para visualizar los datos de prueba y las predicciones
figure;
scatter(X_test(y_test == 0, 1), X_test(y_test == 0, 2), 'x', 'DisplayName', 'Sin Diabetes');
hold on;
scatter(X_test(y_test == 1, 1), X_test(y_test == 1, 2), 'o', 'DisplayName', 'Con Diabetes');
xlabel('Característica 1');
ylabel('Característica 2');
legend('Location', 'Best');
title('Predicciones vs. Datos de Prueba');
grid on;
