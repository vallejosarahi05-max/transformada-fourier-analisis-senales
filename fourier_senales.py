import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parámetros generales
# -----------------------------
fs = 1000            # Frecuencia de muestreo (Hz)
t = np.linspace(-1, 1, fs)

# -----------------------------
# Definición de señales
# -----------------------------

# Pulso rectangular
pulso_rectangular = np.where(np.abs(t) <= 0.2, 1, 0)

# Función escalón
funcion_escalon = np.heaviside(t, 1)

# Señal senoidal
frecuencia = 5       # Frecuencia de la señal (Hz)
senal_senoidal = np.sin(2 * np.pi * frecuencia * t)

# -----------------------------
# Transformada de Fourier
# -----------------------------
def transformada_fourier(senal):
    fft_valores = np.fft.fft(senal)
    fft_valores = np.fft.fftshift(fft_valores)
    frecuencias = np.fft.fftfreq(len(senal), d=1/fs)
    frecuencias = np.fft.fftshift(frecuencias)
    magnitud = np.abs(fft_valores)
    fase = np.angle(fft_valores)
    return frecuencias, magnitud, fase

# Fourier de cada señal
freq_rect, mag_rect, fase_rect = transformada_fourier(pulso_rectangular)
freq_esc, mag_esc, fase_esc = transformada_fourier(funcion_escalon)
freq_sen, mag_sen, fase_sen = transformada_fourier(senal_senoidal)

# -----------------------------
# Gráficas en el dominio del tiempo
# -----------------------------

plt.figure()
plt.plot(t, pulso_rectangular)
plt.title("Pulso rectangular - Dominio del tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, funcion_escalon)
plt.title("Función escalón - Dominio del tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, senal_senoidal)
plt.title("Señal senoidal - Dominio del tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# -----------------------------
# Gráficas en el dominio de la frecuencia
# -----------------------------

plt.figure()
plt.plot(freq_rect, mag_rect)
plt.title("Pulso rectangular - Dominio de la frecuencia")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(freq_esc, mag_esc)
plt.title("Función escalón - Dominio de la frecuencia")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(freq_sen, mag_sen)
plt.title("Señal senoidal - Dominio de la frecuencia")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.show()

# -----------------------------
# Fin del programa
# -----------------------------
