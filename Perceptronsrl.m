% Carga los datos desde el archivo slr.csv
data = readmatrix('slr.csv');

% Divide los datos en características (X) y etiquetas (y)
X = data(:, 1);
y = data(:, 2);

% Define la proporción de entrenamiento (80%) y prueba (20%)
train_ratio = 0.8;
test_ratio = 0.2;

% Genera un índice aleatorio para dividir los datos
rng('default'); % Para reproducibilidad
idx = randperm(length(X));

% Divide los datos en conjuntos de entrenamiento y prueba
num_train = round(train_ratio * length(X));
X_train = X(idx(1:num_train));
y_train = y(idx(1:num_train));
X_test = X(idx(num_train+1:end));
y_test = y(idx(num_train+1:end));

% Entrena un modelo de regresión lineal en los datos de entrenamiento
lm = fitlm(X_train, y_train);

% Evalúa el modelo en los datos de prueba
y_pred = predict(lm, X_test);

% Muestra el rendimiento del modelo en los datos de prueba
MSE = mean((y_pred - y_test).^2);
RMSE = sqrt(MSE);
R2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);

fprintf('Error cuadrático medio (MSE): %.2f\n', MSE);
fprintf('Raíz del error cuadrático medio (RMSE): %.2f\n', RMSE);
fprintf('Coeficiente de determinación (R^2): %.2f\n', R2);

% Visualiza los datos de prueba y las predicciones
scatter(X_test, y_test, 'o', 'DisplayName', 'Datos de prueba');
hold on;
plot(X_test, y_pred, '-r', 'LineWidth', 2, 'DisplayName', 'Predicciones');
xlabel('Cantidad de Terrenos');
ylabel('Precio');
legend('Location', 'Best');
title('Predicciones vs. Datos de Prueba');
grid on;

