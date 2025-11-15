def SGD (f, df, X_old, max_steps, tol_f, tol_x):
    # Dimensione del mini-batch
    k = 3
    # Inizializziamo il dataset
    S_k = np.arange(0,20,1)
    # Per la condizione di uscita
    step_count=0
    # Dichiare un array di 0 di dimensione maxit + 1
    fun_history = np.zeros((max_steps + 1,2))
    fun_history[0,]=X_old

    # Caso che tutto va bene, esce per maxit
    exit_flag = 'maxit'

    current_grad_norm=np.linalg.norm(df(X_old))
    print(f"Norma gradiente corrente: {current_grad_norm}")

    # Controllo se ho superato le iterazioni o se ho raggiunto la tolleranza desiderata
    while step_count < max_steps and current_grad_norm > tol_f :

        # Randomizziamo gli indici del mini-batch
        np.random.shuffle(S_k)

        for i in range (0,len(S_k),k):
            # (Nota: qui useremmo S_k[i:i+k] se df dipendesse dai dati)
            
            # Calcola la direzione (gradiente stocastico)
            grad_stoc = df(X_old)
            p_k = -grad_stoc

            # Backtracking per alpha
            alpha = backtracking(f, df, X_old) # Dobbiamo utilizzarlo? È molto lento

            # Calcolo di X_k+1
            X_new = X_old + alpha * p_k

            # Controllo se sto facendo progressi (tolleranza x)
            if np.linalg.norm(X_new - X_old) < tol_x:
                exit_flag = 'tol_x'
                break

            # Aggiorno per l'iterazione successiva
            X_old = X_new
            step_count += 1
            
            # Aggiungo il valore della funzione in x_k nell'array dei risultati
            fun_history[step_count] = X_new

            # Controllo se ho superato il numero di passi
            if step_count >= max_steps:
                break # Interrompe il 'for' loop

        # Ricalcolo la norma del gradiente
        current_grad_norm=np.linalg.norm(df(X_new))

        if exit_flag == 'tol_x':
            break

    # In caso siamo usciti per tolleranza
    if exit_flag == 'maxit' and current_grad_norm <= tol_f:
        exit_flag = 'tol_f'

    # Tronciamo l'array inizializzato all'inizio
    if step_count<max_steps: 
        fun_history = fun_history[:step_count+1]

    return X_old, step_count, fun_history, exit_flag