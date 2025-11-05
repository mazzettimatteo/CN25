# Metodo del gradiente
def GD (f, df, X_old, maxit, tol_f, tol_x):
    
    count=0
    dim = len(X_old) # Calcolo la dimensione di X_old

    # Dichiare una matrice di 0 di dimensione maxit + 1 per dim
    fun_history = np.zeros((maxit + 1, dim))
    fun_history[0,] = X_old

    exit_flag = 'maxit' # Flag di uscita di default

    current_grad_norm=np.linalg.norm(df(X_old))

    # Controllo se ho superato le iterazioni o se ho raggiunto la tolleranza desiderata
    while count < maxit and current_grad_norm > tol_f:
        # Direzione di discesa
        p_k=-df(X_old)
        
        # Backtracking di alpha
        alpha=backtracking(f, df, X_old, p_k)

        # Calcolo di X_k+1
        X_new = X_old + alpha * p_k

        # Controllo se sto facendo progressi (tolleranza x)
        if np.linalg.norm(X_new-X_old) < tol_x:
            exit_flag = 'tol_x'
            break

        # Ricalcolo la norma del gradiente
        current_grad_norm=np.linalg.norm(df(X_new))

        # Aggiorno per l'iterazione successiva
        X_old = X_new
        count+=1

        # Aggiungo il valore della funzione in x_k nella matrice dei risultati 
        fun_history[count] = X_new

    # Controllo se l'uscita è dovuta a tol_f (norma del gradiente)
    if exit_flag == 'maxit' and current_grad_norm <= tol_f:
        exit_flag = 'tol_f'

    # Rimuovo le righe non utilizzate dalla cronologia
    if count<maxit:
        fun_history = fun_history[:count+1]

    return X_old, count, fun_history, exit_flag