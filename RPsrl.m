% Cargar los datos desde el archivo slr.csv
data = csvread('slr.csv');
X = data(:, 1);  % Tomar solo la primera columna
y = data(:, 2);

% Dividir los datos en conjuntos de entrenamiento y prueba
train_ratio = 0.8;
test_ratio = 0.2;
num_samples = length(X);
num_train = round(train_ratio * num_samples);

X_train = X(1:num_train);
y_train = y(1:num_train);
X_test = X(num_train+1:end);
y_test = y(num_train+1:end);

% Crear y entrenar un modelo de regresión lineal simple
mdl = fitlm(X_train, y_train);

% Realizar predicciones en el conjunto de prueba
y_pred = predict(mdl, X_test);

% Calcular la precisión en el conjunto de prueba (puedes usar otras métricas de regresión)
mse = mean((y_test - y_pred).^2); % Error cuadrático medio

fprintf('Error cuadrático medio en el conjunto de prueba: %.2f\n', mse);

% Graficar resultados
figure;
scatter(X_test, y_test, 'o', 'DisplayName', 'Datos reales');
hold on;
plot(X_test, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'Predicciones');
xlabel('Cantidad de Terrenos');
ylabel('Precio');
legend('Location', 'Best');
title('Predicciones vs. Datos de Prueba');
grid on;
