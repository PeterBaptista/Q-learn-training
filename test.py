import numpy as np
import os

if os.path.exists('resultado.txt'):
    q_table = np.loadtxt('resultado.txt')
    print("Q-table carregada com sucesso!")
else:
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
    print("Criando uma nova Q-table...")