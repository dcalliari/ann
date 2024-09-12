import numpy as np


# Função de ativação Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada da função Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Função para imprimir os cálculos passo a passo, incluindo o backpropagation
def neural_network_step_by_step(
    input_data, W_hidden, W_output, expected_output, learning_rate=0.3
):
    print(f"Entradas: x1 = {input_data[0]}, x2 = {input_data[1]}")

    # 1. Forward pass: Cálculo da camada oculta
    print("\n--- Forward pass: Cálculo da camada oculta ---")
    for i in range(len(W_hidden[0])):
        print(f"\nNeurônio {i+1} da camada oculta:")
        weighted_sum_hidden = (input_data[0] * W_hidden[0][i]) + (
            input_data[1] * W_hidden[1][i]
        )
        print(
            f"Soma ponderada: ({input_data[0]} * {W_hidden[0][i]}) + "
            f"({input_data[1]} * {W_hidden[1][i]}) = {weighted_sum_hidden:.4f}"
        )
        hidden_output = sigmoid(weighted_sum_hidden)
        print(
            f"Aplicando Sigmoid: 1 / (1 + e^-{weighted_sum_hidden:.4f}) = {hidden_output:.4f}"
        )

    # Cálculo conjunto das ativações da camada oculta
    hidden_input = np.dot(input_data, W_hidden)
    hidden_output = sigmoid(hidden_input)
    print(f"\nSaídas da camada oculta (após Sigmoid): {hidden_output}")

    # 2. Forward pass: Cálculo da saída
    print("\n--- Forward pass: Cálculo da saída ---")
    for i in range(len(W_output)):
        print(
            f"Contribuição do neurônio {i+1} da camada oculta para a saída: "
            f"{hidden_output[i]:.4f} * {W_output[i][0]}"
        )

    output_input = np.dot(hidden_output, W_output)
    print(f"Soma ponderada da saída: {output_input[0]:.4f}")
    output = sigmoid(output_input)
    print(f"Aplicando Sigmoid: 1 / (1 + e^-{output_input[0]:.4f}) = {output[0]:.4f}")

    # 3. Calcular o erro na saída
    print("\n--- Cálculo do erro na saída ---")
    error_output = expected_output - output
    print(
        f"Erro da saída (esperado - obtido): {expected_output[0]} - {output[0]:.4f} = {error_output[0]:.4f}"
    )

    # 4. Backpropagation: Cálculo da derivada da sigmoid
    print("\n--- Backpropagation: Derivada da sigmoid na saída ---")
    sigmoid_deriv_output = sigmoid_derivative(output)
    print(
        f"Derivada da sigmoid aplicada à saída: sigmoid_derivada({output[0]:.4f}) = {sigmoid_deriv_output[0]:.4f}"
    )

    # 5. Backpropagation: Gradiente da saída
    print("\n--- Backpropagation: Gradiente da saída ---")
    delta_output = error_output * sigmoid_derivative(output)
    print(
        f"Gradiente da saída (delta_output): {error_output[0]:.4f} * "
        f"sigmoid_derivada({output[0]:.4f}) = {delta_output[0]:.4f}"
    )

    # 6. Propagar o erro para a camada oculta
    print("\n--- Backpropagation: Erro propagado para a camada oculta ---")
    for i in range(len(W_output)):
        print(
            f"Erro propagado para o neurônio {i+1} da camada oculta: "
            f"{delta_output[0]:.4f} * {W_output[i][0]} = {delta_output[0] * W_output[i][0]:.4f}"
        )

    error_hidden = delta_output.dot(W_output.T)
    print(f"Erros propagados para a camada oculta: {error_hidden}")

    # 7. Backpropagation: Gradiente da camada oculta
    print("\n--- Backpropagation: Gradiente da camada oculta ---")
    delta_hidden = error_hidden * sigmoid_derivative(hidden_output)
    for i in range(len(delta_hidden)):
        print(
            f"Gradiente do neurônio {i+1} da camada oculta: "
            f"{error_hidden[i]:.4f} * sigmoid_derivada({hidden_output[i]:.4f}) = {delta_hidden[i]:.4f}"
        )

    # 8. Atualizar os pesos da camada oculta para a saída
    print("\n--- Atualização dos pesos da camada oculta para a saída ---")
    for i in range(len(W_output)):
        print(f"Peso {i+1} antes da atualização: {W_output[i][0]:.4f}")
        W_output[i][0] += learning_rate * hidden_output[i] * delta_output[0]
        print(f"Peso {i+1} após a atualização: {W_output[i][0]:.4f}")

    # 9. Atualizar os pesos da entrada para a camada oculta
    print("\n--- Atualização dos pesos da entrada para a camada oculta ---")
    for i in range(len(W_hidden[0])):
        for j in range(len(W_hidden)):
            print(
                f"Peso da entrada {j+1} para o neurônio {i+1} antes da atualização: {W_hidden[j][i]:.4f}"
            )
            W_hidden[j][i] += learning_rate * input_data[j] * delta_hidden[i]
            print(
                f"Peso da entrada {j+1} para o neurônio {i+1} após a atualização: {W_hidden[j][i]:.4f}"
            )


# Definir entradas e pesos manualmente
input_data = np.array([1, 1])  # Definindo manualmente as entradas
W_hidden = np.array(
    [[-0.424, -0.743, -0.961], [0.358, -0.581, -0.468]]
)  # Pesos para a camada oculta
W_output = np.array(
    [[-0.007], [-0.887], [0.154]]
)  # Pesos da camada oculta para a saída

expected_output = np.array([0])  # Saída esperada

# Executar a rede neural com os valores definidos
neural_network_step_by_step(input_data, W_hidden, W_output, expected_output)
