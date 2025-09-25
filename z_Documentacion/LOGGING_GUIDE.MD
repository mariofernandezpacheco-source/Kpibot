# Sistema de Logging Estructurado - Guía de Uso

## Introducción

El sistema de logging estructurado de π-Bot proporciona:
- **Logging en formato JSON** para análisis automatizado
- **Correlation IDs** para trazar flujos completos
- **Contexto bindeado** por ticker, componente, etc.
- **Decorators automáticos** para funciones críticas
- **Compatibilidad total** con código existente

## Migración Rápida

### 1. Importar el Logger

```python
# ANTES
print("Iniciando proceso...")

# DESPUÉS
from utils.logging_enhanced import get_logger
logger = get_logger(__name__)
logger.info("process_start", message="Iniciando proceso")
```

### 2. Ejecutar Migración Automática

```bash
# Analizar archivos
python scripts/migrate_to_structured_logging.py --dry-run

# Migrar con backup automático
python scripts/migrate_to_structured_logging.py --backup
```

## Patrones de Uso

### Logger Básico

```python
from utils.logging_enhanced import get_logger

logger = get_logger(__name__)

# Diferentes niveles
logger.debug("debug_event", message="Información de debug", detail="extra_info")
logger.info("info_event", message="Información general")
logger.warning("warning_event", message="Advertencia", threshold=0.8)
logger.error("error_event", message="Error occurred", exc_info=True)
```

### Logger con Contexto por Ticker

```python
# Para módulos que procesan múltiples tickers
def process_ticker(ticker):
    # Logger con contexto automático
    ticker_logger = logger.bind(ticker=ticker, timeframe="5mins")
    
    ticker_logger.info("processing_start", message="Iniciando procesamiento")
    ticker_logger.info("signal_generated", signal="BUY", confidence=0.85)
    ticker_logger.info("processing_complete", duration_ms=150)
```

### Correlation IDs para Tracking

```python
from utils.logging_enhanced import with_correlation_id

def execute_complete_workflow(ticker):
    # Todo el flujo tendrá el mismo correlation ID
    with with_correlation_id() as corr_id:
        logger.info("workflow_start", ticker=ticker)
        
        # Llamar a otras funciones - mantienen el mismo corr_id
        data = load_data(ticker)
        prediction = predict(data) 
        result = execute_trade(prediction)
        
        logger.info("workflow_complete", result=result)
        return result
```

### Logging por Componente

```python
from utils.logging_enhanced import with_component

def research_module():
    with with_component("RSH"):
        logger.info("cv_start", n_splits=5, test_size=500)
        # Todos los logs tendrán component="RSH"

def training_module():
    with with_component("TRN"):
        logger.info("training_start", ticker="AAPL")
        # Todos los logs tendrán component="TRN"
```

### Decorators Automáticos

```python
from utils.logging_enhanced import log_function_call

# Logging automático de entrada/salida
@log_function_call(log_args=True, log_result=True)
def calculate_portfolio_value(positions):
    """Función crítica con logging automático"""
    return sum(pos['value'] for pos in positions)

# Solo argumentos
@log_function_call(log_args=True)
def risky_operation(ticker, amount):
    """Función que queremos monitorear"""
    # Se loggea automáticamente entrada, salida y errores
    pass
```

## Configuración

### Archivo YAML

```yaml
# config_/phibot.yaml
logging:
  level: "INFO"                 # DEBUG, INFO, WARNING, ERROR
  structured: true              # Formato JSON
  correlation_id: true          # IDs de trazabilidad
  console_output: true          # Mostrar en consola
  file_output: true            # Guardar en archivo
  log_dir: "logs"
  date_format: "%Y-%m-%d %H:%M:%S"
```

### Personalización

```python
from utils.logging_enhanced import setup_logging

# Reconfigurar logging si necesario
setup_logging()
```

## Ejemplos por Módulo

### DAT - Descarga de Datos

```python
from utils.logging_enhanced import get_logger, with_component

logger = get_logger(__name__)

def download_tickers(tickers):
    with with_component("DAT"):
        logger.info("download_start", 
                   tickers_count=len(tickers),
                   message="Iniciando descarga masiva")
        
        results = []
        for ticker in tickers:
            ticker_logger = logger.bind(ticker=ticker)
            
            try:
                ticker_logger.info("ticker_download_start")
                data = download_single_ticker(ticker)
                
                ticker_logger.info("ticker_download_success", 
                                 rows=len(data),
                                 last_date=data.index[-1].strftime('%Y-%m-%d'))
                results.append(data)
                
            except Exception as e:
                ticker_logger.error("ticker_download_error", 
                                  error_type=type(e).__name__,
                                  exc_info=True)
        
        logger.info("download_complete", 
                   successful=len(results),
                   total=len(tickers))
        
        return results
```

### TRN - Entrenamiento

```python
def train_model(ticker, X, y):
    with with_component("TRN"):
        ml_logger = logger.bind(ticker=ticker)
        
        # Dataset info
        ml_logger.info("dataset_prepared",
                      n_samples=len(X),
                      n_features=X.shape[1],
                      class_distribution=y.value_counts().to_dict())
        
        # Hyperparameters
        params = {'max_depth': 5, 'n_estimators': 100}
        ml_logger.info("hyperparams_set", **params)
        
        # Training
        start_time = time.time()
        model = XGBClassifier(**params)
        model.fit(X, y)
        
        ml_logger.info("training_complete",
                      duration_seconds=time.time() - start_time,
                      train_score=model.score(X, y))
        
        return model
```

### OPS - Operaciones Live

```python
def live_trading_loop():
    session_logger = logger.bind(component="OPS", 
                                session_id=f"session_{datetime.now():%Y%m%d_%H%M}")
    
    session_logger.info("session_start", message="Iniciando sesión de trading")
    
    for ticker in TICKERS:
        with with_correlation_id():  # Nuevo correlation ID por ticker
            ticker_logger = session_logger.bind(ticker=ticker)
            
            try:
                # Process ticker
                signal = generate_signal(ticker)
                ticker_logger.info("signal_generated", 
                                 signal=signal,
                                 timestamp=datetime.utcnow().isoformat())
                
                if signal != 0:
                    result = execute_trade(ticker, signal)
                    ticker_logger.info("trade_executed",
                                     order_id=result['order_id'],
                                     price=result['price'])
                
            except Exception as e:
                ticker_logger.error("ticker_processing_error", 
                                  exc_info=True)
```

## Análisis de Logs

### Formato de Output

Cada log es una línea JSON:

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "logger": "modules.trading",
  "message": "Trade ejecutado exitosamente",
  "event": "trade_executed",
  "correlation_id": "abc123ef",
  "component": "OPS",
  "ticker": "AAPL",
  "price": 150.25,
  "quantity": 100,
  "module": "trading",
  "function": "execute_trade",
  "line": 145
}
```

### Consultas Útiles

```bash
# Filtrar por ticker
grep '"ticker":"AAPL"' logs/phibot.log

# Filtrar por correlation ID (seguir flujo completo)
grep '"correlation_id":"abc123ef"' logs/phibot.log

# Solo errores
grep '"level":"ERROR"' logs/phibot.log

# Métricas de trading
grep '"event":"trade_executed"' logs/phibot.log

# Performance de funciones específicas
grep '"function":"calculate_indicators"' logs/phibot.log
```

### Con herramientas JSON

```bash
# Usando jq para análisis avanzado
cat logs/phibot.log | jq 'select(.ticker == "AAPL" and .event == "trade_executed")'

# Estadísticas por ticker
cat logs/phibot.log | jq -r 'select(.ticker) | .ticker' | sort | uniq -c

# Promedio de duración de operaciones
cat logs/phibot.log | jq 'select(.duration_ms) | .duration_ms' | awk '{sum+=$1; count++} END {print sum/count}'
```

## Troubleshooting

### Problemas Comunes

**1. Logs no aparecen**
```python
# Verificar configuración
from utils.logging_enhanced import get_config
config = get_config()
print(f"Log level: {config.logging.level}")
print(f"Console output: {config.logging.console_output}")
```

**2. Formato incorrecto**
```python
# Forzar reconfiguración
from utils.logging_enhanced import setup_logging
setup_logging()
```

**3. Archivos de log muy grandes**
- Los archivos rotan automáticamente (50MB por defecto)
- Se mantienen 10 backups por defecto
- Configurar en `logging.log_rotation_mb` y `logging.backup_count`

### Debug del Sistema

```python
# Test completo del sistema
python utils/logging_enhanced.py

# Test específico
python -c "
from utils.logging_enhanced import get_logger
logger = get_logger('test')
logger.info('test_event', message='Test message')
print('Logging test completado')
"
```

## Migración Paso a Paso

### 1. Módulo por Módulo

Recomendación: migrar un módulo a la vez para verificar funcionalidad.

```bash
# Solo DAT_Data_download.py
python scripts/migrate_to_structured_logging.py \
  --root-dir . \
  --include-pattern "**/DAT_*.py" \
  --dry-run
```

### 2. Verificar Migración

```python
# En cada módulo migrado, añadir al principio:
from utils.logging_enhanced import get_logger
logger = get_logger(__name__)

# El script automático añade esto, pero verificar manualmente
```

### 3. Testing

```bash
# Test del sistema completo
python -m pytest tests/test_logging_system.py -v

# Test integración con módulo migrado
python your_migrated_module.py
```

## Mejores Prácticas

### Do's

- **Usar events descriptivos**: `"model_training_start"` mejor que `"info"`
- **Incluir contexto relevante**: ticker, timeframe, timestamp
- **Usar correlation IDs** para flujos complejos
- **Logging estructurado** siempre en JSON para análisis
- **Bindear contexto** para módulos que procesan múltiples elementos

### Don'ts

- **No loggear información sensible**: contraseñas, tokens
- **No logs en loops intensivos** sin throttling
- **No usar print() mezclado** con logging estructurado
- **No logs sin contexto**: incluir siempre información relevante

### Performance

```python
# Throttling para loops intensivos
last_log_time = 0

for i, item in enumerate(huge_list):
    # Solo loggear cada 100 items o cada 5 segundos
    if i % 100 == 0 or time.time() - last_log_time > 5:
        logger.info("processing_progress", 
                   processed=i, 
                   total=len(huge_list))
        last_log_time = time.time()
```

## Conclusión

El sistema de logging estructurado proporciona:
- **Trazabilidad completa** de operaciones
- **Análisis automatizado** de performance y errores  
- **Debugging eficiente** con correlation IDs
- **Monitoreo en producción** con métricas estructuradas

La migración es gradual y mantiene compatibilidad total con el código existente.