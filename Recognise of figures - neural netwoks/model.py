import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Funkcja sprawdzająca, czy liczba jest pierwsza
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# Generowanie danych treningowych dla parzystości i pierwszości
def generate_data(limit=1000):
    X = np.array(range(2, limit + 2)).reshape(-1, 1)
    y_parity = np.array([1 if x % 2 == 0 else 0 for x in X.flatten()])  # 1 dla parzystych, 0 dla nieparzystych
    y_prime = np.array([1 if is_prime(x) else 0 for x in X.flatten()])  # 1 dla pierwszych, 0 dla złożonych
    y = np.column_stack((y_parity, y_prime))  # Łączymy etykiety parzystości i pierwszości w jedną macierz
    return X, y

# Sprawdzenie poprawności generowanych danych
X, y = generate_data(1000)
print("Przykładowe dane wejściowe i etykiety:")
for i in range(len(X)):
    print(f"Liczba: {X[i][0]}, Parzystość: {'parzysta' if y[i][0] == 1 else 'nieparzysta'}, Pierwszość: {'pierwsza' if y[i][1] == 1 else 'złożona'}")

# Tworzenie modelu sieci neuronowej
def create_model():
    model = Sequential([
        Dense(128, input_dim=1, activation='relu'),  # Zwiększenie liczby neuronów
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='sigmoid')  # Dwa wyjścia: parzystość i pierwszość
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Trenowanie modelu
X, y = generate_data(1000)
model = create_model()
model.fit(X, y, epochs=5000, batch_size=10, verbose=1)  # Zwiększenie liczby epok

# Testowanie modelu na przykładowych liczbach po trenowaniu
test_numbers = [6, 24, 15, 23, 2, 100, 101, 13]  # Liczba 13 dodana do testów
print("\nWyniki predykcji dla testowych liczb:")
for num in test_numbers:
    prediction = model.predict(np.array([[num]]))
    is_even = "parzysta" if prediction[0][0] > 0.5 else "nieparzysta"
    is_prime = "pierwsza" if prediction[0][1] > 0.5 else "złożona"
    print(f"Liczba: {num}, Parzystość: {is_even}, Pierwszość: {is_prime}")

# Zapisanie modelu do pliku
model.save('number_classification_model.keras')
print("\nModel zapisany jako number_classification_model.keras")
