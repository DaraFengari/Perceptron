% Cargar los datos desde el archivo slr.csv
data = csvread('slr.csv');
X = data(:, 1);  % Tomar solo la primera columna
y = data(:, 2);

% Inicializar variables para almacenar precisiones
mse_loo = zeros(size(y));
mse_lko = zeros(size(y));

% Leave-One-Out (LOO)
for i = 1:length(y)
    X_train_loo = X;
    y_train_loo = y;
    X_train_loo(i) = [];
    y_train_loo(i) = [];
    
    X_test_loo = X(i);
    y_test_loo = y(i);
    
    % Crear y entrenar un modelo de regresión lineal simple para LOO
    mdl_loo = fitlm(X_train_loo, y_train_loo);
    
    % Realizar predicciones en el conjunto de prueba
    y_pred_loo = predict(mdl_loo, X_test_loo);
    
    % Calcular el error cuadrático medio para LOO
    mse_loo(i) = (y_test_loo - y_pred_loo)^2;
end

mse_loo_average = mean(mse_loo);
fprintf('Error cuadrático medio usando Leave-One-Out (LOO): %.2f\n', mse_loo_average);

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

    X_train_lko = X(train_indices_lko);
    y_train_lko = y(train_indices_lko);
    
    X_test_lko = X(test_indices_lko);
    y_test_lko = y(test_indices_lko);
    
    % Crear y entrenar un modelo de regresión lineal simple para LKO
    mdl_lko = fitlm(X_train_lko, y_train_lko);
    
    % Realizar predicciones en el conjunto de prueba
    y_pred_lko = predict(mdl_lko, X_test_lko);
    
    % Calcular el error cuadrático medio para LKO
    mse_lko(test_indices_lko) = (y_test_lko - y_pred_lko).^2;
end

mse_lko_average = mean(mse_lko);
fprintf('Error cuadrático medio usando Leave-K-Out (LKO) con k=%d: %.2f\n', k, mse_lko_average);

% Graficar resultados
figure;
scatter(X, y, 'o', 'DisplayName', 'Datos reales');
hold on;
plot(X, predict(mdl, X), 'r-', 'LineWidth', 2, 'DisplayName', 'Predicciones Original');
xlabel('Cantidad de Terrenos');
ylabel('Precio');
legend('Location', 'Best');
title('Predicciones vs. Datos de Prueba');
grid on;

