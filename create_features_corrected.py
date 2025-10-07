def create_features_simple(series):
    """
    Función SÚPER SIMPLE - solo features temporales + algunos lags básicos
    MANTIENE exactamente las mismas filas que las series originales
    """
    df = pd.DataFrame(index=series.index)
    
    # === FEATURES TEMPORALES (solo estas, que nunca fallan) ===
    df['year'] = series.index.year
    df['month'] = series.index.month
    df['day'] = series.index.day
    df['dayofweek'] = series.index.dayofweek
    df['quarter'] = series.index.quarter
    
    # Features cíclicas (si no rompen)
    try:
        df['month_sin'] = np.sin(2 * np.pi * series.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * series.index.month / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * series.index.dayofweek / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * series.index.dayofweek / 7)
    except:
        pass  # Si rompe, no las agregamos
    
    # Lags básicos (si no rompen)
    try:
        df['lag_1'] = series.shift(1)
        df['lag_5'] = series.shift(5)
    except:
        pass  # Si rompe, no los agregamos
    
    # TARGET para train/test
    df['target'] = series.shift(-1)
    
    # SIN dropna() - mantenemos exactamente las mismas filas
    return df

# ========================================
# CÓDIGO PARA EL NOTEBOOK (USAR ESTO):
# ========================================

# Crear features SIMPLES usando las variables ya existentes
features_data = {}

for ticker, train_data, test_data in [
    ('NVDA', nvda_train, nvda_test),
    ('AMD', amd_train, amd_test), 
    ('INTC', intel_train, intel_test)
]:
    print(f"Procesando {ticker}...")
    
    # Train features
    train_features = create_features_simple(train_data)
    # QUITAR última fila (target NaN por shift(-1))
    train_features = train_features[:-1]
    
    X_train = train_features.drop('target', axis=1).fillna(0)
    y_train = train_features['target']  # Ya no tiene NaN
    
    # Test features  
    test_features = create_features_simple(test_data)
    # QUITAR última fila (target NaN por shift(-1))
    test_features = test_features[:-1]
    
    X_test = test_features.drop('target', axis=1).fillna(0)
    y_test = test_features['target']  # Ya no tiene NaN
    
    features_data[ticker] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    
    print(f"  X_train: {X_train.shape}, y_train: {len(y_train)}")
    print(f"  X_test: {X_test.shape}, y_test: {len(y_test)}")

print("✅ Features creadas exitosamente!")

# Ejemplo de uso CORRECTO - respetando los datos del notebook:

# 1. Usar las variables ya definidas en el notebook
# nvda_train, amd_train, intel_train (ya particionadas en 80/20)

# 2. Crear features solo para el conjunto de ENTRENAMIENTO  
# features_df = create_features(nvda_train)  # NO close['NVDA']

# 3. Luego usar para predicción en test set
# print(f"Features creadas: {list(features_df.columns)}")
# print(f"Shape: {features_df.shape}")

# ========================================
# CÓDIGO PARA USAR EN EL NOTEBOOK:
# ========================================

"""
# Crear features para cada ticker USANDO LOS DATOS YA PARTICIONADOS
features_data = {}

for ticker, train_data in [('NVDA', nvda_train), ('AMD', amd_train), ('INTC', intel_train)]:
    print(f"Procesando {ticker}...")
    
    # Crear features usando SOLO los datos de entrenamiento
    features_df = create_features(train_data)
    
    # Ya están listas las features, sin necesidad de partición adicional
    X_train = features_df.drop('target', axis=1)
    y_train = features_df['target']
    
    # Para test set, usar los datos de test correspondientes
    if ticker == 'NVDA':
        test_data = nvda_test
    elif ticker == 'AMD':
        test_data = amd_test
    else:  # INTC
        test_data = intel_test
    
    # Crear features para test (sin target)
    test_features_df = create_features(test_data)
    X_test = test_features_df.drop('target', axis=1)
    y_test = test_features_df['target']
    
    features_data[ticker] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
"""
